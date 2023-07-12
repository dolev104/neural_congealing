# Code based on GANgealing's code (https://github.com/wpeebles/gangealing/blob/main/applications/propagate_to_images.py)
"""
This script propagates edits in atlas space to the original images and visualizes average image.
To apply edits, first extract the average image using evaluate_model.py.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import math
import os

from torchvision import transforms
from tqdm import tqdm

from datasets.dataset import AtlasDataset
from models.model import AtlasModel
from models.spatial_transformers.spatial_transformer import SpatialTransformer
from utils_atlas.utils_atlas_base import upload_trained_model_and_data, apply_griddata
from utils_atlas.vis_tools.helpers import load_dense_label, images2grid, save_image, make_grid, splat_points, save_video
import torch.nn.functional as F


def write(image_batch, folder_name, filenames_suffix, input_nrow=None, pad_value=-1.0):
    # Saves image_batch to disk
    nrow = int(np.ceil(math.sqrt(image_batch.size(0)))) if input_nrow is None else input_nrow
    # Save grid of output images:
    save_image(image_batch, f'{args.out}/{folder_name}_grid.{filenames_suffix}', normalize=True, range=(0, 1), nrow=nrow, pad_value=pad_value, padding=3)
    if args.save_individual_images:  # Save each output image individually in a new folder:
        os.makedirs(f'{args.out}/{folder_name}', exist_ok=True)
        for i, image in enumerate(image_batch):
            save_image(image.unsqueeze_(0), f'{args.out}/{folder_name}/{i:03}.{filenames_suffix}')


@torch.inference_mode()
def get_images(dataset):
    input_images = []
    input_images_to_model = []
    for i in range(dataset.number_of_images):
        curr_img_input_to_model, _, _ = dataset.get_sample_data(i, allow_return_all=False)
        curr_img, _, _ = dataset.get_img_data(i)
        input_images.append(curr_img)
        input_images_to_model.append(curr_img_input_to_model)
    input_images = torch.stack(input_images, dim=0)  # [num_imgs, 3, resy, resx]
    input_images_to_model = torch.stack(input_images_to_model, dim=0)
    flip_indices = dataset.images_state if dataset.config["ext_horizontal_flips"] else torch.zeros(len(input_images))

    return input_images, input_images_to_model, flip_indices


def get_label_data(args, number_of_images, device, use_splat=True):
    if args.label_path is not None:
        points, colors, alpha_channel, label = load_dense_label(args.label_path, use_griddata=(not use_splat),
                                                                resolution=args.output_resolution, device=device)
        points = points.repeat(number_of_images, 1, 1)
        points_normalized = SpatialTransformer.normalize(points, args.output_resolution, label.shape[-2])
        if label.shape[-2] != args.output_resolution:
            points = SpatialTransformer.convert(points, label.shape[-2], args.output_resolution).round().long()
            label = torch.nn.functional.interpolate(
                label.float(), scale_factor=args.output_resolution / label.size(2), mode='bilinear')
    else:
        points = points_normalized = colors = alpha_channel = label = None
    return alpha_channel, colors, label, points, points_normalized


def expand_rank3_batch(tensor, batch_size):
    if tensor is not None:
        return tensor.expand(batch_size, -1, -1)


@torch.inference_mode()
def make_visuals(args, dataset, model, device):
    # (1) Upload images:
    input_images, input_images_to_model, flip_indices = get_images(dataset)
    input_images = input_images.to(device)
    input_images_to_model = input_images_to_model.to(device)

    # (2) Congealed (aligned) images:
    print('Congealing (aligning) images...')
    transformations_params, congealed_images, _ = \
        model.process_batch(input_images_to_model, im_output_res=args.output_resolution, return_images=True)
    _, _, _, all_final_grids = transformations_params

    if args.output_resolution != input_images.shape[-2]:
        input_images = torch.nn.functional.interpolate(
            input_images, scale_factor=args.output_resolution / input_images.shape[-2], mode='bilinear')
        input_images_to_model = torch.nn.functional.interpolate(
            input_images_to_model, scale_factor=args.output_resolution / input_images_to_model.shape[-2], mode='bilinear')
    if args.video_frames_path is None:
        write(input_images, 'input_images', args.filenames_suffix)

    # (3) Edit Propagation/ Dense Correspondence:
    print(f'Propagating {args.label_path} to images...')
    alpha_channels, colors, edit_image, points, points_normalized = get_label_data(args, input_images.size(0), device, use_splat=args.use_splat)
    points_normalized = expand_rank3_batch(points_normalized, input_images.size(0))
    colors = expand_rank3_batch(colors, input_images.size(0))
    alpha_channels = expand_rank3_batch(alpha_channels, input_images.size(0))
    save_image(edit_image, f'{args.out}/image_label_to_propagate.png')

    # Compute where points_normalized lie in the original unaligned images:
    upoints = model.uncongeal_points(input_images_to_model, points_normalized, samp_grid=all_final_grids, normalize_input_points=False)
    # We need to flip the points wherever flips were chosen during training so they are correctly overlaid on the original, unflipped images:
    upoints[:, :, 0] = torch.where(flip_indices.view(-1, 1).to(device) == 1, args.output_resolution - 1 - upoints[:, :, 0], upoints[:, :, 0])

    if args.apply_gray_background:
        input_images_for_edit = transforms.Grayscale()(input_images).cpu()
    else:
        input_images_for_edit = input_images

    if not args.use_splat:
        out_indices = torch.cartesian_prod(torch.arange(input_images.shape[-2]), torch.arange(input_images.shape[-1])).float().numpy()
        upoints_cpu_np = upoints.cpu().numpy()
        colors_alpha_cpu_np = torch.cat((colors, alpha_channels), dim=-1).cpu().numpy()
        propagated_images = []
        for i in tqdm(range(input_images.shape[0])):
            curr_image = input_images_for_edit[i].unsqueeze(0).cpu()
            edited_image = apply_griddata(curr_image, upoints_cpu_np[i], colors_alpha_cpu_np[i], out_indices)
            propagated_images.append(edited_image)
        propagated_images = torch.cat(propagated_images, dim=0)
    else:
        propagated_images = splat_points(input_images_for_edit, upoints, sigma=args.sigma, opacity=args.opacity,
                                         colorscale='plasma', colors=colors, alpha_channel=alpha_channels)

    if args.video_frames_path is not None:
        frames_grids = []
        for i in tqdm(range(propagated_images.shape[0])):
            orig_and_edited_frame = torch.stack((input_images[i].cpu(), propagated_images[i].cpu()), dim=0)
            image_grid = images2grid(orig_and_edited_frame, nrow=2, normalize=True, range=(0, 1), scale_each=False,
                                     pad_value=args.grid_pad_value)
            frames_grids.append(image_grid)
        save_video(frames_grids, fps=args.vid_fps, out_path=f'{args.out}/input_and_edited_video.mp4')
        save_video(propagated_images, fps=args.vid_fps, out_path=f'{args.out}/edited_video.mp4', input_is_tensor=True)
    else:
        write(propagated_images, 'propagated', args.filenames_suffix)
        images_orig_and_annotated = make_grid(torch.cat((input_images.cpu(), propagated_images.cpu()), dim=0),
                                              nrow=input_images.shape[0], normalize=False, padding=3, pad_value=args.grid_pad_value)
        save_image(images_orig_and_annotated, f'{args.out}/image_orig_and_annotated_grid.{args.filenames_suffix}')

        if args.create_video:
            # Create video of edit fading in and out
            output_video = create_fade_in_out_video(args, input_images, propagated_images)
            save_video(output_video, fps=args.fps, out_path=f'{args.out}/edits_fade_in_fade_out.mp4')

    # (4) Annotated Average Image in atlas space:
    average_image = congealed_images.mean(dim=0, keepdim=True).cpu()
    # Obtain the atlas saliency
    _, atlas_saliency = model.get_atlas_params()
    atlas_saliency = atlas_saliency[None, None, ...]
    resized_atlas_saliency = torch.nn.functional.interpolate(
        atlas_saliency, scale_factor=args.output_resolution / atlas_saliency.size(2), mode='bilinear')

    checkerboard_image = transforms.ToTensor()(Image.open('data/checkerboard_gray_256.png').convert("RGB"))[None, ...]
    checkerboard_image = torch.nn.functional.interpolate(
        checkerboard_image, scale_factor=args.output_resolution / checkerboard_image.size(2), mode='bilinear')
    average_image_ckerboard = average_image * resized_atlas_saliency.cpu() + checkerboard_image * (1 - resized_atlas_saliency.cpu())

    label = edit_image.float().cpu()
    label_alpha = label[:, 3:4, ...]
    annotated_average = average_image_ckerboard * (1 - label_alpha) + label_alpha * label[:, :3]
    save_image(annotated_average, f'{args.out}/average_image_annotated.{args.filenames_suffix}')

    atlas_avg_and_annotated_concat = torch.cat((average_image_ckerboard.cpu(), annotated_average.cpu()), dim=0)
    atlas_avg_and_annotated = make_grid(atlas_avg_and_annotated_concat, nrow=1, normalize=False, padding=3, pad_value=args.grid_pad_value)
    save_image(atlas_avg_and_annotated, f'{args.out}/average_image_orig_and_annotated_grid.{args.filenames_suffix}')

    # Also visualize average image in image space
    average_image_naive = input_images.mean(dim=0, keepdim=True).cpu()
    save_image(average_image_naive, f'{args.out}/average_image_space.{args.filenames_suffix}')

    print(f'All output images can be found at {args.out}')


def create_fade_in_out_video(args, input_images, propagated_images):
    out_frame = []
    nrow = int(np.ceil(math.sqrt(input_images.size(0))))
    all_propagated_images = propagated_images.to(input_images.dtype).to(device)

    for frame_ix in tqdm(range(args.length)):
        # Below we smoothly interpolate alpha between 0 and 1 using cosine annealing:
        alpha = 1 - 0.5 * (1 + torch.cos(torch.tensor(math.pi * frame_ix / (args.length - 1)))).to(device)

        semi_edits = torch.lerp(input_images, all_propagated_images, alpha.view(1, 1, 1, 1))
        image_grid = images2grid(semi_edits, nrow=nrow, normalize=True, range=(0, 1), scale_each=False,
                                 pad_value=args.grid_pad_value)
        out_frame.append(image_grid)

    if args.extend_on_edited > 0:
        image_grid = images2grid(all_propagated_images, nrow=nrow, normalize=True, range=(0, 1), scale_each=False,
                                 pad_value=args.grid_pad_value)
        last_frames = [image_grid] * args.extend_on_edited
        out_frame.extend(last_frames)

    reverse = [out_frame[-1 - i] for i in range(len(out_frame))]
    output_video = out_frame + reverse
    return output_video


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use pre-trained atlas congealing mode to propagate edits.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="path to model checkpoint.")
    # Visualization hyperparameters:
    parser.add_argument("--label_path", type=str, required=True, help='Path to a dense label in congealed space, '
                                                                      'formatted as an RGBA image')
    parser.add_argument("--use_splat", action='store_true', help='If specified, will use splat2d instead of griddata '
                                                                 'to propagate the edit')
    parser.add_argument("-s", "--sigma", default=0.9, type=float, help='Relevant if use_splat is True')
    parser.add_argument("-o", "--opacity", default=1., type=float, help='Relevant if use_splat is True')

    parser.add_argument("--save_individual_images", action='store_true',
                        help='If specified, saves all output images to disk individually '
                             '(default: saves grids of output images)')
    parser.add_argument("--out", type=str, default='visuals_and_edits', help='directory where results will be saved')
    parser.add_argument('--filenames_suffix', type=str, default="png", help="jpg | png")
    parser.add_argument('--grid_pad_value', type=int, default=1, help="pad value for image grids created by "
                                                                      "make_grid ([0 1]).")
    parser.add_argument('--output_resolution', type=int, default=512, help="output resolution for edited images.")
    parser.add_argument('--apply_gray_background', action='store_true', help="If specified, only the edit will be "
                                                                             "colored in the edited images.")
    # Options for editing video frames (original training set is representative frames of the video)
    parser.add_argument("--video_frames_path", type=str, default=None, help='path to main folder of video frames.')
    parser.add_argument("--vid_fps", type=int, default=10, help='FPS of saved edited video.')
    # Options for fade-in fade-out video. Ignored if video_frames_path is not None
    parser.add_argument("--create_video", action='store_true', help='If specified, will create a video with the edit '
                                                                    'fading in and out of original images.')
    parser.add_argument("--length", type=int, default=170, help='The number of frames to generate.')
    parser.add_argument("--extend_on_edited", type=int, default=50, help='The number of frames to keep on the '
                                                                         'final edited images.')
    parser.add_argument("--fps", type=int, default=90, help='FPS of saved video')

    args = parser.parse_args()

    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    # load trained model and data
    _, config, dataset, model = upload_trained_model_and_data(args.checkpoint_path, AtlasModel, AtlasDataset, device)
    if args.video_frames_path is not None:  # trained on representative frames. Applying trained model on all video frames.
        config["data_folder"] = args.video_frames_path
        dataset = AtlasDataset(config, device)

    image_set_name = str(Path(config["data_folder"]).name)
    now = datetime.now()
    args.out = f"{args.out}/{now.strftime('%Y-%m-%d_%H-%M-%S')}_{image_set_name}"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        make_visuals(args, dataset, model, device)

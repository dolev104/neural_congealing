from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import torchvision
from matplotlib import cm
from scipy.interpolate import griddata
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image

from datasets.dataset import AtlasDataset
from models.model import AtlasModel
from models.spatial_transformers.spatial_transformer import SpatialTransformer
from utils_atlas.logger import DataLogger
from utils_atlas.utils_atlas_base import calculate_fw_matrix, upload_trained_model_and_data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_model(args):
    _, config, dataset, model = upload_trained_model_and_data(args.checkpoint_path, AtlasModel, AtlasDataset, device)
    if args.video_frames_path is not None:  # trained on representative frames. Applying trained model on all video frames.
        config["data_folder"] = args.video_frames_path
        dataset = AtlasDataset(config, device)
        args.output_fps = args.vid_fps
    config["pca_visualize_with_global_minmax"] = args.pca_visualize_with_global_minmax
    config["use_griddata"] = args.use_griddata
    image_resolution = config["image_resolution"]

    print("Evaluating model...")
    # evaluation
    logger = DataLogger(config, dataset, model, device, log_to_wandb=False, eval=True)
    evaluation_data = logger.evaluate_and_log(dataset, model, log_to_wandb=False, sim_only_bootstrap=(not model.with_flow), eval_only_mode=True)
    imgs_pca_logging_dict = logger.imgs_pca_logging_dict

    image_set_name = str(Path(config["data_folder"]).name)
    now = datetime.now()
    output_folder = Path(f"{args.output_dir}/{now.strftime('%Y-%m-%d_%H-%M-%S')}_{image_set_name}")
    output_folder.mkdir(parents=True, exist_ok=True)

    # save the checkpoint path
    with open(f"{output_folder}/checkpoint_path.txt", "w") as fd:
        fd.write(args.checkpoint_path)

    if args.video_frames_path is None:
        save_image(evaluation_data[f"congealing/congealing_images_grid"],
                   output_folder / f"congealing_res_mappings_grid.{args.filenames_suffix}")
    torchvision.io.write_video(str(output_folder / f"mapping_images_video.mp4"),
                               evaluation_data[f"congealing/congealing_images_video"].permute(0, 2, 3, 1) * 255,
                               fps=args.output_fps, video_codec="libx264")

    # original images, saliencies and keys
    curr_path = output_folder / "input_images"
    curr_path.mkdir(parents=True, exist_ok=True)
    curr_path_orig = curr_path / "input_set"
    curr_path_orig.mkdir(parents=True, exist_ok=True)
    curr_path_masked_orig_orig_sal = curr_path / "input_set_orig_sal_masked"
    curr_path_masked_orig_orig_sal.mkdir(parents=True, exist_ok=True)
    curr_path_masked_orig_atlas_sal = curr_path / "input_set_atlas_sal_masked"
    curr_path_masked_orig_atlas_sal.mkdir(parents=True, exist_ok=True)

    curr_path_pca_keys = output_folder / "image_set_keys_pca"
    curr_path_pca_keys.mkdir(parents=True, exist_ok=True)
    curr_path_sal = output_folder / "image_set_saliencies_masks"
    curr_path_sal.mkdir(parents=True, exist_ok=True)

    # congealed images (masked and unmasked)
    curr_path_sim = output_folder / "congealed_images_STNsim"
    curr_path_sim.mkdir(parents=True, exist_ok=True)
    if model.with_flow:
        final_congealed_images_folder = output_folder / "congealed_images"
        curr_path_congealed = final_congealed_images_folder / "congealed_images_checkerboard"
        curr_path_congealed.mkdir(parents=True, exist_ok=True)
        curr_path_w = final_congealed_images_folder / "congealed_images_white_masked"
        curr_path_w.mkdir(parents=True, exist_ok=True)
        curr_path_o = final_congealed_images_folder / "congealed_images_unmasked"
        curr_path_o.mkdir(parents=True, exist_ok=True)

    # transformed keys and saliency
    curr_path_ckp = output_folder / "congealed_keys_pca"
    curr_path_ckp.mkdir(parents=True, exist_ok=True)
    curr_path_cs = output_folder / "congealed_saliency_masks"
    curr_path_cs.mkdir(parents=True, exist_ok=True)

    # for propagating atlas saliency to images
    atlas_saliency = evaluation_data["atlas_vis/atlas_saliency"]
    resized_atlas_saliency = evaluation_data["atlas_vis/resized_atlas_saliency"]
    i, j = torch.where(atlas_saliency > -1)  # take all points
    points_atlas_sal = torch.stack([j, i], -1)  # (P, 2); points are stored in (x, y) format
    colors_atlas_sal = atlas_saliency[i, j]  # (1, P, 3), [0, 1]
    atlas_resy = atlas_saliency.shape[-1]
    points_normalized_atlas_sal = SpatialTransformer.normalize(points_atlas_sal, image_resolution, atlas_resy)

    atlas_saliency = atlas_saliency[None, None, ...]
    _, _, im_resy, im_resx = dataset.all_images.shape
    checkerboard_image = evaluation_data["checkerboard_image"]
    checkerboard = transforms.Resize(atlas_saliency.shape[-1])(checkerboard_image)

    # save keys pca of input images
    for p in range(logger.pca_n_components):
        save_image(imgs_pca_logging_dict[f"pca_vis_orig_imgs_keys/pca_ch{p + 1}"],
                   curr_path_pca_keys / f"pca_vis_orig_imgs_keys_ch{p + 1}.png")
    if logger.pca_n_components > 3:
        save_image(imgs_pca_logging_dict["pca_vis_orig_imgs_keys/pca_ch234"],
                   curr_path_pca_keys / "pca_vis_orig_imgs_keys_ch234.png")
    keys_pca_all = imgs_pca_logging_dict[f"pca_vis_orig_imgs_keys/pca_ch234_sep"].float()

    all_orig_imgs = []
    all_input_saliencies = []
    all_input_saliencies_overlay = []
    all_congealed_checkerboard = []

    for i, img in enumerate(dataset.all_images):
        input_image = img[None, ...]
        input_saliency = dataset.imgs_saliency_maps[i][None, ...]
        all_orig_imgs.append(input_image.cpu())

        save_image(input_image, curr_path_orig / f"original_image_{i:03d}.{args.filenames_suffix}")
        save_image(input_saliency, curr_path_sal / f"saliency_image_{i:03d}.{args.filenames_suffix}")

        # image masked with initial saliency mask
        resized_input_saliency = transforms.Resize(im_resx)(input_saliency).cpu()
        masked_img_orig_sal_checkerboard = input_image.cpu() * resized_input_saliency + (1 - resized_input_saliency) * checkerboard_image.cpu()
        save_image(masked_img_orig_sal_checkerboard, curr_path_masked_orig_orig_sal / f"masked_img_orig_sal_checkerboard_{i:03d}.{args.filenames_suffix}")
        all_input_saliencies_overlay.append(masked_img_orig_sal_checkerboard.cpu())

        curr_sampling_grid = evaluation_data["all_sampling_grids"][i][None, ...]

        # show masked image using atlas saliency (need to transform atlas saliency to image space)
        if model.with_flow:
            curr_input_image = evaluation_data["all_input_images"][i][None, ...]
            upoints_atlas_sal = model.uncongeal_points(curr_input_image, points_normalized_atlas_sal[None, ...],
                                                       samp_grid=curr_sampling_grid.detach(), normalize_input_points=False).cpu().numpy()[0]
            atlas_sal_img_space = griddata(upoints_atlas_sal, colors_atlas_sal.cpu(), (logger.out_indices[:, 1], logger.out_indices[:, 0]), method='linear')
            atlas_sal_img_space = transforms.ToTensor()(atlas_sal_img_space.reshape(im_resy, im_resx))[None, ...]
            atlas_sal_img_space[atlas_sal_img_space.isnan()] = 0.
        else:
            curr_affine_params = evaluation_data["all_affine_params"][i][None, ...]
            fw_matrix = calculate_fw_matrix(curr_affine_params)
            atlas_sal_img_space = model.apply_bw_warp(resized_atlas_saliency, fw_matrix, flow=None, padding_mode='zeros',
                                                      img_size=resized_atlas_saliency.size()).cpu()
        if dataset.images_state[i] == 1:  # need to flip mask to match original orientation
            atlas_sal_img_space = torch.flip(atlas_sal_img_space, dims=[-1])
        masked_img_atlas_sal_checkerboard = input_image.cpu() * atlas_sal_img_space + (1 - atlas_sal_img_space) * checkerboard_image.cpu()
        save_image(masked_img_atlas_sal_checkerboard,  curr_path_masked_orig_atlas_sal / f"masked_img_atlas_sal_checkerboard_{i:03d}.{args.filenames_suffix}")
        all_input_saliencies.append(resized_input_saliency.expand(-1, 3, -1, -1).cpu())

        # congealed images
        curr_sim_congealed = evaluation_data["all_unmasked_congealed_imgs_sim"][i][None, ...].cpu()
        save_image(curr_sim_congealed, curr_path_sim / f"sim_congealed_image_{i:03d}.{args.filenames_suffix}")
        if model.with_flow:
            curr_congealed = evaluation_data["all_congealed_imgs_checkerboard"][i][None, ...]
            save_image(curr_congealed, curr_path_congealed / f"congealed_image_{i:03d}_checkerboard.{args.filenames_suffix}")
            all_congealed_checkerboard.append(curr_congealed.cpu())

            final_congealed_image = evaluation_data["all_unmasked_congealed_imgs"][i][None, ...]
            save_image(final_congealed_image, curr_path_o / f"congealed_image_{i:03d}_unmasked.{args.filenames_suffix}")
            curr_white_masked = final_congealed_image * resized_atlas_saliency + \
                                (1 - resized_atlas_saliency) * (0.55 * final_congealed_image + 0.45 * torch.ones_like(final_congealed_image))
            save_image(curr_white_masked, curr_path_w / f"congealed_image_{i:03d}_w.{args.filenames_suffix}")
        else:
            all_congealed_checkerboard.append(curr_sim_congealed)

        # congealed keys and saliencies
        save_image(evaluation_data["all_congealed_saliencies"][i], curr_path_cs / f"congealed_saliency_{i:03d}.{args.filenames_suffix}")
        keys_pca = keys_pca_all[i][None, ...]
        if dataset.images_state[i] == 1:  # horizontally flip
            keys_pca = torch.flip(keys_pca, dims=[-1])
        if model.with_flow:
            congealed_keys_pca = model.apply_bw_warp(keys_pca.to(device), None, flow=curr_sampling_grid,
                                                     padding_mode='border', img_size=keys_pca.size())
        else:
            congealed_keys_pca = model.apply_bw_warp(keys_pca.to(device), curr_sampling_grid, flow=None,
                                                     padding_mode='border', img_size=keys_pca.size())
        congealed_keys_checkerboard = congealed_keys_pca * atlas_saliency + (1 - atlas_saliency) * checkerboard
        save_image(congealed_keys_checkerboard, curr_path_ckp / f"congealed_keys_pca_checkerboard_{i:03d}.{args.filenames_suffix}")

    if args.video_frames_path is None:
        # create image grids (original images, initial masks and final congealed images)
        all_imgs_w_orig_sal = torch.cat(all_orig_imgs + all_input_saliencies + all_congealed_checkerboard, dim=0)
        all_imgs_w_orig_sal_grid = make_grid(all_imgs_w_orig_sal, nrow=dataset.number_of_images, normalize=False,
                                             padding=3, pad_value=args.imgs_grid_pad_value)
        save_image(all_imgs_w_orig_sal_grid, output_folder / f"congealing_res_grid_orig_sal.{args.filenames_suffix}")
        all_imgs_w_overlay_sal = torch.cat(all_orig_imgs + all_input_saliencies_overlay + all_congealed_checkerboard, dim=0)
        all_imgs_w_overlay_sal_grid = make_grid(all_imgs_w_overlay_sal, nrow=dataset.number_of_images, normalize=False,
                                                padding=3, pad_value=args.imgs_grid_pad_value)
        save_image(all_imgs_w_overlay_sal_grid, output_folder / f"congealing_res_grid_overlay_sal.{args.filenames_suffix}")

        all_imgs_and_congealed = torch.cat(all_orig_imgs + all_congealed_checkerboard, dim=0)
        all_imgs_and_congealed_grid = make_grid(all_imgs_and_congealed, nrow=dataset.number_of_images, normalize=False,
                                                padding=3, pad_value=args.imgs_grid_pad_value)
        save_image(all_imgs_and_congealed_grid, output_folder / f"congealing_res_grid.{args.filenames_suffix}")

    # atlas visualizations:
    # average image in atlas space
    curr_path = output_folder / "atlas_vis"
    curr_path.mkdir(parents=True, exist_ok=True)
    atlas_space_average_image = evaluation_data["atlas_vis/atlas_space_average_image_unmasked"]
    save_image(atlas_space_average_image, curr_path / f"atlas_space_average_image_unmasked.{args.filenames_suffix}")
    atlas_space_average_image_checkerboad = evaluation_data["atlas_vis/atlas_space_average_image"]
    save_image(atlas_space_average_image_checkerboad, curr_path / f"atlas_space_average_image.{args.filenames_suffix}")

    # atlas saliency
    save_image(atlas_saliency.squeeze(), curr_path / f"atlas_saliency.{args.filenames_suffix}")
    # PCA on atlas keys using PCA comp calculated on input image set
    atlas_pca234_of_set = evaluation_data["atlas_vis/atlas_keys_pca_of_set"][:, :, 1:4].permute(2, 0, 1)[None, ...]
    save_image(atlas_pca234_of_set, curr_path / f"pca_of_set_ch234_atlas_keys.{args.filenames_suffix}")
    atlas_keys_checkerboard = atlas_pca234_of_set * atlas_saliency.cpu() + (1 - atlas_saliency.cpu()) * checkerboard.cpu()
    save_image(atlas_keys_checkerboard, curr_path / f"pca_of_set_ch234_atlas_keys_checkerboard.{args.filenames_suffix}")
    save_image(transforms.ToTensor()(cm.get_cmap('viridis')(evaluation_data["atlas_vis/atlas_keys_pca_of_set"][:, :, 0]))[None, ...],
               curr_path / f"pca_of_set_ch1_atlas_keys.png")
    # PCA on atlas keys
    atlas_pca234 = evaluation_data["atlas_vis/atlas_keys_pca"][:, :, 1:4].permute(2, 0, 1)[None, ...]
    save_image(atlas_pca234, curr_path / f"pca_ch234_atlas_keys.{args.filenames_suffix}")
    atlas_keys_checkerboard = atlas_pca234 * atlas_saliency.cpu() + (1 - atlas_saliency.cpu()) * checkerboard.cpu()
    save_image(atlas_keys_checkerboard, curr_path / f"pca_ch234_atlas_keys_checkerboard.{args.filenames_suffix}")
    save_image(transforms.ToTensor()(cm.get_cmap('viridis')(evaluation_data["atlas_vis/atlas_keys_pca"][:, :, 0]))[None, ...],
               curr_path / f"pca_ch1_atlas_keys.png")

    # summary atlas vis
    atlas_sal_resized = resized_atlas_saliency.expand(-1, 3, -1, -1)
    atlas_keys_checkerboard = transforms.Resize(atlas_space_average_image_checkerboad.shape[-1])(atlas_keys_checkerboard).expand(-1, 3, -1, -1)
    atlas_vis_concat = torch.cat((atlas_space_average_image_checkerboad.cpu(), atlas_sal_resized.cpu(), atlas_keys_checkerboard.cpu()), dim=0)
    atlas_vis_grid_checkerboard = make_grid(atlas_vis_concat, nrow=1, normalize=False, padding=3, pad_value=args.imgs_grid_pad_value)
    save_image(atlas_vis_grid_checkerboard, curr_path / f"atlas_vis_grid.{args.filenames_suffix}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True, help="path to model checkpoint.")
    parser.add_argument('--output_dir', type=str, default="evaluation_output", help="main folder for evaluation outputs")
    parser.add_argument('--filenames_suffix', type=str, default="png", help="jpg | png")
    parser.add_argument('--use_griddata', action='store_false', help="If specified, will use splat instead of griddata for propagating edits.")
    parser.add_argument('--pca_visualize_with_global_minmax', action='store_false', help="If specified, normalize each channel separately.")
    parser.add_argument('--imgs_grid_pad_value', type=int, default=1, help="pad value for image grids using make_grid ([0 1])")
    parser.add_argument('--output_fps', type=int, default=1, help="fps for video showing the complete results")
    # Options for evaluating on all video frames (original training set is representative frames of the video)
    parser.add_argument("--video_frames_path", type=str, default=None, help='path to main folder of video frames.')
    parser.add_argument("--vid_fps", type=int, default=10, help='FPS of saved evaluation video.')

    args = parser.parse_args()

    evaluate_model(args)

# Code taken from https://github.com/ShirAmir/dino-vit-features/blob/main/inspect_similarity.py
# modified to run on a set of images instead of a pair
import yaml

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from datasets.dataset import AtlasDataset
from models.vit_extractor_model import ViTExtractor
from utils_atlas.utils_atlas_base import l2_loss


def chunk_cosine_sim(x, y, similarity_metric='cosine'):
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :param similarity_metric: [cosine|l2] metric for calculating similarities between descriptors.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        if similarity_metric == 'l2':
            result_list.append(l2_loss(token[0], y[0]))  # 1xtxd'
        else:
            result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def show_similarity_interactive(args, config):
    # extract descriptors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dino_extractor = ViTExtractor(
        model_type=config["dino_model_type"], stride=args.dino_stride, device=device
    )
    config["atlas_resolution"] = 0
    config["dino_keys_layer"] = args.dino_keys_layer
    config["use_masks"] = False
    dataset = AtlasDataset(config, device, dino_extractor=dino_extractor)
    reference_image = args.ref_image_idx
    num_patches, load_size = dino_extractor.num_patches, dino_extractor.load_size
    patch_size = dino_extractor.model.patch_embed.patch_size
    stride = dino_extractor.stride[0]

    all_keys = dataset.imgs_dino_keys

    # plot
    number_of_images = dataset.number_of_images
    num_cols = int(np.ceil(np.sqrt(2 * number_of_images)))
    num_rows = int(np.ceil(2 * number_of_images / num_cols)) + 1
    if (num_rows - 1) * num_cols >= (number_of_images * 2):
        num_rows -= 1
    elif num_rows * (num_cols - 1) >= (number_of_images * 2):
        num_cols -= 1
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    [axi.set_axis_off() for axi in ax.ravel()]
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    visible_patches = []
    radius = 4
    fig_fontsize = 5
    ax[0, 0].imshow(dataset.all_images[reference_image].permute(1, 2, 0))
    ax[0, 0].set_title(f'Reference Image {reference_image}', fontsize=fig_fontsize)
    j = 1
    for i in range(number_of_images):
        if i == reference_image:
            continue
        ax[j // num_cols, j % num_cols].imshow(dataset.all_images[i].permute(1, 2, 0))
        ax[j // num_cols, j % num_cols].set_title('Image ' + str(i), fontsize=fig_fontsize)
        j += 1

    emb_dize = all_keys.shape[1]
    descs_reference = all_keys[reference_image].permute(1, 2, 0).reshape(1, 1, -1, emb_dize)  # [1, 1, t, d]
    num_patches = all_keys.shape[2], all_keys.shape[3]
    new_H = patch_size / stride * (load_size[0] // patch_size - 1) + 1
    new_W = patch_size / stride * (load_size[1] // patch_size - 1) + 1

    # start interactive loop
    fig.suptitle('Select a point on the left image. \n Right click to stop.', fontsize=5)
    plt.draw()
    pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
    starting_point = number_of_images
    while len(pts) == 1:
        all_sim_images = []
        y_coor, x_coor = int(pts[0, 1]), int(pts[0, 0])
        y_descs_coor = int(new_H / load_size[0] * y_coor)
        x_descs_coor = int(new_W / load_size[1] * x_coor)
        raveled_desc_idx = num_patches[1] * y_descs_coor + x_descs_coor
        curr_descs_reference = descs_reference[:, :, [raveled_desc_idx]]

        # reset previous marks
        for patch in visible_patches:
            patch.remove()
        visible_patches = []

        # draw chosen point on original image
        center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                  (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
        patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
        ax[0, 0].add_patch(patch)
        visible_patches.append(patch)

        j = 0
        for i in range(number_of_images):
            # get and draw current similarities
            curr_desc = all_keys[i].permute(1, 2, 0).reshape(1, 1, -1, emb_dize)  # [1, 1, t, d]

            curr_similarities = chunk_cosine_sim(curr_descs_reference, curr_desc,
                                                 similarity_metric=args.similarity_metric)  # [1, 1, num_tokens_imA, num_tokens_imB]
            if args.similarity_metric == 'l2':
                curr_similarities = curr_similarities.mean(dim=-1)
                curr_similarities = (curr_similarities - curr_similarities.min()) / (
                            curr_similarities.max() - curr_similarities.min())
                curr_similarities = 1. - curr_similarities
            curr_similarities = curr_similarities.reshape(num_patches)

            all_sim_images.append(curr_similarities)
            curr_idx = starting_point + j
            ax[curr_idx // num_cols, curr_idx % num_cols].imshow(curr_similarities.cpu().numpy(), cmap='jet')

            j += 1

        plt.draw()

        # get input point from user
        fig.suptitle('Select a point on the top-left image', fontsize=5)
        plt.draw()
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


if __name__ == "__main__":
    # Code taken from https://github.com/ShirAmir/dino-vit-features/blob/main/inspect_similarity.py
    # modified to run on a set of images instead of a pair
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        help="Config path",
    )
    parser.add_argument("--ref_image_idx", type=int, default=0)
    parser.add_argument("--similarity_metric", type=str, default='cosine', help="l2 | cosine")
    parser.add_argument("--dino_keys_layer", type=int, default=11)
    parser.add_argument("--dino_stride", type=int, default=4, help="dino stride")

    parser.add_argument("--folder_path", type=str, required=True, help="path to folder containing image set")

    args = parser.parse_args()

    config_path = args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    config["data_folder"] = args.folder_path
    config["ext_horizontal_flips"] = False
    with torch.no_grad():
        show_similarity_interactive(args, config)

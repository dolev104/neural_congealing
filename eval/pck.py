# Code built upon GANgealing's (https://github.com/wpeebles/gangealing/blob/ffa6387c7ffd3f7de76bdc693dc2272e274e9bfd/applications/pck.py)
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import argparse

from torchvision.utils import make_grid
from PIL import Image
import termcolor
from tqdm import tqdm

from datasets.dataset import AtlasDataset
from models.model import AtlasModel
from utils_atlas.utils_atlas_base import upload_trained_model_and_data
from utils_atlas.vis_tools.helpers import batch_overlay

NUMBER_OF_CUB_SUBSETS = 14  # number of CUB subsets used for numeric evaluation (see paper for details)


def images2grid(images, **grid_kwargs):
    # images should be (N, C, H, W)
    grid = make_grid(images, **grid_kwargs)
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return out


def get_pairs_kps(args, path):
    keypoints_path = f'{path}/keypoints.pt'
    orig_pairs_path = f'{path}/pairs.pt'
    pairs_path = f'{path}/pairs_indices_in_set.pt'
    thresh_path = f'{path}/pck_thresholds.pt'
    inverse_ops_path = f'{path}/inverse_coordinates.pt'
    permutation_path = f'{path}/permutation.pt'
    if not args.dataset == "cub":  # relevant for SPair categories only
        orig_pairs = torch.load(orig_pairs_path)
        fixed_pairs = torch.load(pairs_path)
        thresholds = torch.load(thresh_path)
        inverse_ops = torch.load(inverse_ops_path)
    else:
        orig_pairs, fixed_pairs, thresholds, inverse_ops = None, None, None, None
    keypoints = torch.load(keypoints_path)
    mirror_permutation = torch.load(permutation_path)

    return orig_pairs, fixed_pairs, keypoints, thresholds, inverse_ops, mirror_permutation


def get_pck(args, data_path, config, dataset, model, device):
    if args.vis_transfer:
        pairs_to_save_counter = 0
        imgs_a_to_save = []
        imgs_b_to_save = []
        imgs_a_gt_pnts = []
        imgs_b_gt_pnts = []
        imgs_b_pred_pnts = []

    images = dataset.all_images

    # get key points for this run dataset
    orig_pairs, pairs, kps, thresholds, inverse_ops, mirror_permutation = get_pairs_kps(args, f"{data_path}/pck")
    if args.dataset != 'cub':
        transfer_both_ways = False  # both ways are already built in
    else:
        transfer_both_ways = True
        pairs = torch.combinations(torch.arange(dataset.number_of_images), r=2)  # [all_possible_pairs_with_no_repetition, 2]
        orig_pairs = pairs

    correct = 0.
    key_points_seen = 0
    print("number of pairs:", pairs.shape[0] * (1 + transfer_both_ways))
    for i in tqdm(range(pairs.shape[0])):
        idxA, idxB = pairs[i]
        idxA, idxB = idxA.item(), idxB.item()
        imgA, imgB = images[idxA].to(device), images[idxB].to(device)
        # get gt keypoints
        ixA, ixB = orig_pairs[i]
        ixA, ixB = ixA.item(), ixB.item()
        gt_kpsA, gt_kpsB = kps[ixA].to(device), kps[ixB].to(device)
        if inverse_ops is not None and thresholds is not None:
            thresh_a = thresholds[ixA]
            scale_a = inverse_ops[ixA, 2]
            thresh_b = thresholds[ixB]
            scale_b = inverse_ops[ixB, 2]

        if gt_kpsA.size(-1) == 3:  # (x, y, visibility):
            visible_kps = gt_kpsA[..., 2:3] * gt_kpsB[..., 2:3]  # Create a mask to ignore non-visible key points
            gt_kpsA, gt_kpsB = gt_kpsA[..., :2].clone().unsqueeze(0), gt_kpsB[..., :2].clone().unsqueeze(0)  # Remove visibility information

        if args.vis_transfer and pairs_to_save_counter < args.vis_number_of_pairs:
            imgs_a_to_save.append(imgA.clone())
            imgs_b_to_save.append(imgB.clone())
            imgs_a_gt_pnts.append(gt_kpsA.clone())
            imgs_b_gt_pnts.append(gt_kpsB.clone())
        imgA = imgA.unsqueeze(0)
        imgB = imgB.unsqueeze(0)
        permutation = mirror_permutation  # shape [num_pnts,]

        if config["ext_horizontal_flips"]:  # need to check orientation of images
            imA_state = dataset.images_state[idxA]
            imB_state = dataset.images_state[idxB]
            imgA, gt_kpsA, imgB, gt_kpsB = update_permutation(imgA, gt_kpsA, imA_state, imgB, gt_kpsB, imB_state,
                                                              permutation)

        est_kpsB = model.transfer_points(imgA, imgB, gt_kpsA, padding_mode=config["stn_padding_mode_im"])

        if args.vis_transfer and pairs_to_save_counter < args.vis_number_of_pairs:
            est_kpsB_orig = est_kpsB.clone()
            if config["ext_horizontal_flips"] and imB_state == 1:
                # bring back pnts to original orientation if needed:
                est_kpsB_orig[:, :, 0] = imgB.size(-1) - 1 - est_kpsB_orig[:, :, 0]
            imgs_b_pred_pnts.append(est_kpsB_orig)
            pairs_to_save_counter += 1

        if thresholds is None:  # threshold used for CUB
            imgB_thresh = torch.tensor(max(imgB.size(-2), imgB.size(-1)), device=device)
        else:  # threshold used for SPair-71K categories
            imgB_thresh = (scale_b * thresh_b).to(device)
        thresholdB = args.alpha * imgB_thresh
        # Compute accuracies at the specified alpha threshold
        err_A2B = (est_kpsB - gt_kpsB).norm(dim=-1).unsqueeze_(-1)
        correct_batch_A2B = err_A2B <= thresholdB
        correct += correct_batch_A2B.mul(visible_kps).sum(dim=(0, 1))

        if transfer_both_ways:  # for CUB. For SPair, pairs are already listed both ways.
            est_kpsA = model.transfer_points(imgB, imgA, gt_kpsB, padding_mode=config["stn_padding_mode_im"])
            if thresholds is None:  # threshold used for CUB
                imgA_thresh = torch.tensor(max(imgA.size(-2), imgA.size(-1)), device=device)
            else:  # threshold used for SPair-71K categories
                imgA_thresh = (scale_a * thresh_a).to(device)
            thresholdA = args.alpha * imgA_thresh
            err_B2A = (est_kpsA - gt_kpsA).norm(dim=-1).unsqueeze_(-1)
            correct_batch_B2A = err_B2A <= thresholdA
            correct += correct_batch_B2A.mul(visible_kps).sum(dim=(0, 1))

        key_points_seen += visible_kps.sum() * (1 + transfer_both_ways)

    # Normalize by the number of pairs observed times the number of key points per-image:
    pck_alpha = correct / key_points_seen

    if args.vis_transfer:
        out_vis = (imgs_a_to_save, imgs_b_to_save, imgs_a_gt_pnts, imgs_b_gt_pnts, imgs_b_pred_pnts)
    else:
        out_vis = None

    return pck_alpha, out_vis


def update_permutation(imgA, gt_kpsA, img_stateA, imgB, gt_kpsB, img_stateB, permutation=None):
    def update_state(img, pnts, state):
        upd_img = img.clone()
        upd_pnts = pnts.clone()
        if state == 1:
            upd_img = upd_img.flip(3, )
            upd_pnts[:, :, 0] = upd_img.size(-1) - 1 - upd_pnts[:, :, 0]
        return upd_img, upd_pnts
    imgA, gt_kpsA = update_state(imgA, gt_kpsA, img_stateA)
    imgB, gt_kpsB = update_state(imgB, gt_kpsB, img_stateB)
    if (img_stateA == 1 and img_stateB == 0) or (img_stateA == 0 and img_stateB == 1):
        gt_kpsA = gt_kpsA.clone()
        gt_kpsA = gt_kpsA[:, permutation]
    return imgA, gt_kpsA, imgB, gt_kpsB


def vis_transfer_atlas(output_folder, imgs_a_to_save, imgs_b_to_save, imgs_a_gt_pnts, imgs_b_gt_pnts,
                       imgs_b_pred_pnts, num_to_vis=8):
    number_of_batches = int(np.ceil(len(imgs_a_to_save) / num_to_vis))
    out_path = f'{output_folder}/transfers'

    for i in range(number_of_batches):
        curr_imgsA = torch.stack(imgs_a_to_save[i * num_to_vis:(i + 1) * num_to_vis], dim=0)
        curr_imgsB = torch.stack(imgs_b_to_save[i * num_to_vis:(i + 1) * num_to_vis], dim=0)
        curr_gt_kpsA = torch.cat(imgs_a_gt_pnts[i * num_to_vis:(i + 1) * num_to_vis], dim=0)
        curr_gt_kpsB = torch.cat(imgs_b_gt_pnts[i * num_to_vis:(i + 1) * num_to_vis], dim=0)
        curr_est_kpsB = torch.cat(imgs_b_pred_pnts[i * num_to_vis:(i + 1) * num_to_vis], dim=0)

        nrow = len(curr_imgsA)
        imgs = torch.cat([curr_imgsA, curr_imgsB]).cpu()
        kps = torch.cat([curr_gt_kpsA, curr_est_kpsB]).cpu()
        out = batch_overlay(imgs, kps, None, out_path, unique_color=True, size=10)
        grid = images2grid(out, nrow=nrow, normalize=True, range=(0, 255))
        grid_path = f'{out_path}/transfer_grid_{i}.png'
        Image.fromarray(grid).save(grid_path)

        kps = torch.cat([curr_gt_kpsA, curr_gt_kpsB]).cpu()
        out = batch_overlay(imgs, kps, None, out_path, unique_color=True, size=10)
        grid = images2grid(out, nrow=nrow, normalize=True, range=(0, 255))
        grid_path = f'{out_path}/transfer_grid_{i}_gt.png'
        Image.fromarray(grid).save(grid_path)
        print(f'Saved visualization to {grid_path}')


def run_pck_eval(args, device):
    if args.dataset == "cub":
        data_path = f"data/cub_subsets/cub_subset_{args.cub_subset}"
    else:
        set_name = args.dataset if args.dataset != "cat_rigid" else 'cat'
        data_path = f'data/spair_sets/spair_{set_name}_test'

    print("specified path:", data_path)
    # load trained model and config
    checkpoint_path, config, dataset, model = load_trained_model_and_set(args, data_path, device)

    pck, out_vis = get_pck(args, data_path, config, dataset, model, device)
    pck_string = format_pck_string(pck, args.alpha)
    print(pck_string)

    if out_vis is not None:
        image_set_name = str(Path(config["data_folder"]).name) if args.dataset != "cat_rigid" else "spair_cat_test_rigid"
        now = datetime.now()
        output_folder = f"{args.output_dir}/{now.strftime('%Y-%m-%d_%H-%M-%S')}_{image_set_name}"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        # save the checkpoint path
        with open(f"{output_folder}/checkpoint_path.txt", "w") as fd:
            fd.write(checkpoint_path)

        file1 = open(
            f"{output_folder}/pck_{np.round(pck.item() * 100, 3)}_perc_for_alpha_{format(args.alpha, '.2f')}",
            "a")
        file1.close()

        imgs_a_to_save, imgs_b_to_save, imgs_a_gt_pnts, imgs_b_gt_pnts, imgs_b_pred_pnts = out_vis
        vis_transfer_atlas(output_folder, imgs_a_to_save, imgs_b_to_save, imgs_a_gt_pnts, imgs_b_gt_pnts,
                           imgs_b_pred_pnts)

    return pck


def run_pck_eval_multi(args, device):
    total_pck = 0.
    for set_num in range(NUMBER_OF_CUB_SUBSETS):
        args.cub_subset = set_num
        pck = run_pck_eval(args, device)
        total_pck += pck
    print('*' * 100)
    total_pck = total_pck / NUMBER_OF_CUB_SUBSETS
    pck_string = format_pck_string(total_pck, args.alpha)
    print(f"Average results for all subsets: {pck_string}")


def format_pck_string(pck_val, alpha):
    return termcolor.colored(f'PCK-Transfer@{alpha}: {np.round(pck_val.item() * 100, 2)}%', 'blue')


def load_trained_model_and_set(args, data_path, device):
    prefix = f"{args.dataset}_" if args.dataset != 'cub' else ''
    checkpoint_path = f"{data_path}/{prefix}checkpoint_epoch_8000.pt"
    _, config, dataset, trained_model = \
        upload_trained_model_and_data(checkpoint_path, AtlasModel, AtlasDataset, device, upload_masks=False)
    model = trained_model.get_mapping_model()
    return checkpoint_path, config, dataset, model


if __name__ == "__main__":
    # Code built upon GANgealing's (https://github.com/wpeebles/gangealing/blob/ffa6387c7ffd3f7de76bdc693dc2272e274e9bfd/applications/pck.py)
    parser = argparse.ArgumentParser(description='Compute PCK-Transfer.')
    parser.add_argument("--dataset", type=str, default='cub', help="Name of Spair category "
                                                                   "['cat' | 'cat_rigid' | 'dog' | 'bicycle'] "
                                                                   "or CUB ['cub']."
                                                                   "'cat_rigid' refers to the more rigid result of "
                                                                   "'cat' category of SPair.")
    parser.add_argument("--cub_subset", type=int, default=0, help="Number of cub subset (14 subsets were randomly "
                                                                   "sampled for eval, see paper for details)")
    parser.add_argument("--all_cub", action='store_true', help="If specified when dataset == 'cub', will run evaluation "
                                                               "on all subsets and output the average result across all sets.")

    parser.add_argument('--alpha', type=float, default=0.1, help="Threshold at which to evaluate PCK")
    parser.add_argument('--output_dir', type=str, default="numeric_evaluations", help="")

    parser.add_argument('--vis_transfer', action='store_true', help='If specified, saves visualization of key point transfers')
    parser.add_argument('--vis_number_of_pairs', type=int, default=8, help='Number of pairs to visualize')

    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        if args.dataset == 'cub' and args.all_cub:
            run_pck_eval_multi(args, device)
        else:
            run_pck_eval(args, device)

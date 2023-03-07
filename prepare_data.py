# Code taken from GANgealing (https://github.com/wpeebles/gangealing/blob/1f64782a35e0b623501c3e87f05240ca1cd82af6/prepare_data.py)
# Same preprocessing of SPair-71K and CUB as in GANgealing, modified to match Neural Congealing files convention
import argparse
import shutil
from pathlib import Path

from PIL import Image
from torchvision.datasets.utils import download_file_from_google_drive, extract_archive, download_and_extract_archive
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import json
import os

from glob import glob
from utils_atlas.CUB_data_utils import square_bbox, perturb_bbox, acsm_crop


# When an image is mirrored, any key points with left/right distinction need to be swapped.
# These are the permutations of key point indices that accomplishes this:
CUB_PERMUTATION = [0, 1, 2, 3, 4, 5, 10, 11, 12, 9, 6, 7, 8, 13, 14]
SPAIR_PERMUTATIONS = {
    'bicycle': [0, 1, 3, 2, 4, 5, 7, 6, 8, 10, 9, 11],
    'cat': [1, 0, 3, 2, 5, 4, 7, 6, 8, 10, 9, 12, 11, 13, 14],
    'dog': [1, 0, 3, 2, 5, 4, 6, 7, 8, 10, 9, 12, 11, 13, 14, 15],
}

NUMBER_OF_CUB_SUBSETS = 14  # number of CUB subsets used for numeric evaluation (see paper for details)


def download_cub_metadata(to_path):
    # Downloads some metadata so we can use image pre-processing consistent with ACSM for CUB
    # and get the lists of subset images for constructing the same 14 subsets from the dataset as shown in the paper
    cub_metadata_folder = f'{to_path}/cub_metadata'
    if not os.path.isdir(cub_metadata_folder):
        print('Downloading metadata used to form ACSM\'s CUB validation set, together with CUB subsets filenames')
        cub_metadata_file_id = "1Upa-5mjMqHZGTHuDEk7mZCCMUs7To232"
        download_file_from_google_drive(cub_metadata_file_id, to_path)
        zip_path = f'{cub_metadata_folder}.zip'
        shutil.move(f'{to_path}/{cub_metadata_file_id}', zip_path)
        extract_archive(zip_path, remove_finished=True)
    else:
        print(f'Found pre-existing CUB metadata folder at {cub_metadata_folder}')


def download_cub(to_path):
    # Downloads the CUB-200-2011 dataset
    cub_dir = f'{to_path}/CUB_200_2011'
    if not os.path.isdir(cub_dir):
        print(f'Downloading CUB_200_2011 to {to_path}')
        cub_url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
        download_and_extract_archive(cub_url, cub_dir, remove_finished=True)
    else:
        print('Found pre-existing CUB directory')
    return f'{cub_dir}/CUB_200_2011'


def download_spair(to_path):
    # Downloads and extracts the SPair-71K dataset
    spair_dir = f'{to_path}/SPair-71k'
    if not os.path.isdir(spair_dir):
        print(f'Downloading SPair-71k to {to_path}')
        spair_url = 'http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz'
        download_and_extract_archive(spair_url, to_path, remove_finished=True)
    else:
        print('Found pre-existing SPair-71K directory')
    return spair_dir


def border_pad(img, target_res, resize=True, to_pil=True):
    original_width, original_height = img.size
    if original_height <= original_width:
        if resize:
            img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        half_height = (target_res - height) / 2
        int_half_height = int(half_height)
        lh = int_half_height
        rh = int_half_height + (half_height > int_half_height)
        img = np.pad(img, mode='edge', pad_width=[(lh, rh), (0, 0), (0, 0)])
    else:
        if resize:
            img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.ANTIALIAS)
        width, height = img.size
        img = np.asarray(img)
        half_width = (target_res - width) / 2
        int_half_width = int(half_width)
        lw = int_half_width
        rw = int_half_width + (half_width > int_half_width)
        img = np.pad(img, mode='edge', pad_width=[(0, 0), (lw, rw), (0, 0)])
    if to_pil:
        img = Image.fromarray(img)
    return img


def center_crop(img, target_res):
    # From official StyleGAN2 create_lsun method:
    img = np.asarray(img)
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
    img = Image.fromarray(img, 'RGB')
    img = img.resize((target_res, target_res), Image.ANTIALIAS)
    return img


def cub_crop(img, target_res, bbox):
    # This function mimics ACSM's pre-processing used for the CUB dataset (up to image resampling and padding color)
    img = np.asarray(img)
    img = acsm_crop(img, bbox, 0, border=True)
    return Image.fromarray(img).resize((target_res, target_res), Image.ANTIALIAS)


def resize_and_convert(img, size, method, bbox=None):
    if method == 'border':
        img = border_pad(img, size)
    elif method == 'center':
        img = center_crop(img, size)
    elif method == 'none':
        pass
    elif method == 'cub_crop':
        img = cub_crop(img, size, bbox)
    else:
        raise NotImplementedError

    return img


def process_images_and_save(out_path, files, method, size, bboxes, image_format):
    output_path_images = f'{out_path}/images'
    Path(output_path_images).mkdir(parents=True, exist_ok=True)
    num_files = len(files)
    print(f'Found {num_files} files for {out_path}')
    print(f'Example file being loaded: {files[0]}')
    files = [(i, file, bbox) for i, (file, bbox) in enumerate(zip(files, bboxes))]
    print("Saving images...")
    for img_file in tqdm(files):
        i, file, bbox = img_file
        img = Image.open(file).convert('RGB')
        out_img = resize_and_convert(img, size, method, bbox)
        # save image
        image_filename = f'{output_path_images}/img_{i:03d}.{image_format}'
        out_img.save(image_filename)


def load_image_folder_and_process(path, method, size, out_path, image_format):
    files = sorted(list(glob(f"{path}/*.jpeg")) + list(glob(f"{path}/*.jpg")) + list(glob(f"{path}/*.png")))
    bboxes = [None] * len(files)  # This means no bounding boxes are used
    process_images_and_save(out_path, files, method, size, bboxes, image_format)


def preprocess_kps_pad(kps, img_width, img_height, size):
    # Once an image has been pre-processed via border (or zero) padding,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the border-padded (or zero-padded) image.
    kps = kps.clone()
    scale = size / max(img_width, img_height)
    kps[:, [0, 1]] *= scale
    if img_height < img_width:
        new_h = int(np.around(size * img_height / img_width))
        offset_y = int((size - new_h) / 2)
        offset_x = 0
        kps[:, 1] += offset_y
    elif img_width < img_height:
        new_w = int(np.around(size * img_width / img_height))
        offset_x = int((size - new_w) / 2)
        offset_y = 0
        kps[:, 0] += offset_x
    else:
        offset_x = 0
        offset_y = 0
    kps *= kps[:, 2:3]  # zero-out any non-visible key points
    return kps, offset_x, offset_y, scale


def preprocess_kps_box_crop(kps, bbox, size):
    # Once an image has been pre-processed via a box crop,
    # the location of key points needs to be updated. This function applies
    # that pre-processing to the key points so they are correctly located
    # in the cropped image.
    kps = kps.clone()
    kps[:, 0] -= bbox[0] + 1
    kps[:, 1] -= bbox[1] + 1
    w = 1 + bbox[2] - bbox[0]
    h = 1 + bbox[3] - bbox[1]
    assert w == h
    kps[:, [0, 1]] *= size / float(w)
    return kps


def load_CUB_keypoints(path):
    names = ['img_index', 'kp_index', 'x', 'y', 'visible']
    landmarks = pd.read_table(path, header=None, names=names, delim_whitespace=True, engine='python')
    landmarks = landmarks.to_numpy().reshape((11788, 15, 5))[..., [2, 3, 4]]  # (num_images, num_kps, 3)
    landmarks = torch.from_numpy(landmarks).float()
    return landmarks


def extract_cub_subset_acsm_data(files, filenames, kps, b, curr_images_list_file, out_path, size):
    with open(curr_images_list_file, "r") as fd:
        subset_filenames = fd.read().splitlines()
    output_path_pck = f'{out_path}/pck'
    Path(output_path_pck).mkdir(parents=True, exist_ok=True)
    subset_files = []
    bboxes = []
    kps_out = []
    for file in subset_filenames:
        curr_file_index = filenames.index(file)  # find index of current file in filenames

        curr_file = files[curr_file_index]
        curr_kps = kps[curr_file_index]
        curr_b = b[curr_file_index]

        x1, y1, x2, y2 = curr_b[0, 0]
        bbox = np.array([x1[0, 0], y1[0, 0], x2[0, 0], y2[0, 0]]) - 1
        bbox = perturb_bbox(bbox, 0.05, 0)
        bbox = square_bbox(bbox)
        bboxes.append(bbox)
        kps_out.append(preprocess_kps_box_crop(curr_kps, bbox, size))
        subset_files.append(curr_file)
    bboxes = np.stack(bboxes)
    kps_out = torch.stack(kps_out)
    torch.save(kps_out, f'{output_path_pck}/keypoints.pt')
    # When an image is mirrored horizontally, the designation between key points with a left versus right distinction
    # needs to be swapped. This is the permutation of CUB key points which accomplishes this swap:
    torch.save(CUB_PERMUTATION, f'{output_path_pck}/permutation.pt')  # TODO: could save just once for all subsets
    assert bboxes.shape[0] == len(subset_filenames)
    return subset_files, bboxes


def load_acsm_data_and_process(path, metadata_path='data/cub_metadata', method='cub_crop', size=256, out_path=None, image_format='png'):
    from scipy.io import loadmat
    mat_path = f'{metadata_path}/acsm_val_cub_cleaned.mat'
    mat = loadmat(mat_path)
    files = [f'{path}/images/{file[0]}' for file in mat['images']['rel_path'][0]]
    # These are the indices retained by ACSM (others are filtered):
    indices = [i[0, 0] - 1 for i in mat['images']['id'][0]]
    kps = load_CUB_keypoints(f'{path}/parts/part_locs.txt')[indices]
    b = mat['images']['bbox'][0]

    filenames = [f'CUB_200_2011/images/{file[0]}' for file in mat['images']['rel_path'][0]]
    # upload subsets' original paths and process each subset's data
    for set_idx in tqdm(range(NUMBER_OF_CUB_SUBSETS)):
        curr_out_path = f'{out_path}/cub_subset_{set_idx}'
        curr_images_list_file = f'{metadata_path}/cub_subset_{set_idx}_original_paths.txt'
        subset_files, bboxes = extract_cub_subset_acsm_data(files, filenames, kps, b, curr_images_list_file, curr_out_path, size)
        process_images_and_save(curr_out_path, subset_files, method, size, bboxes, image_format)


def load_spair_pck_data(path, category, split, size, out_path):
    output_path_pck = f'{out_path}/pck'
    Path(output_path_pck).mkdir(parents=True, exist_ok=True)
    pairs = sorted(glob(f'{path}/PairAnnotation/{split}/*{category}.json'))

    files_no_repetition = []  # images' names in set, with no repetitions
    pairs_set_indices = []  # original index in set with no repetitions
    thresholds = []
    inverse = []
    category_anno = list(glob(f'{path}/ImageAnnotation/{category}/*.json'))[0]
    with open(category_anno) as f:
        num_kps = len(json.load(f)['kps'])
    print(f'Number of SPair key points for {category} <= {num_kps}')
    kps = []
    blank_kps = torch.zeros(num_kps, 3)
    for pair in pairs:
        with open(pair) as f:
            data = json.load(f)
        assert category == data["category"]
        assert data["mirror"] == 0
        source_fn = f'{path}/JPEGImages/{category}/{data["src_imname"]}'
        target_fn = f'{path}/JPEGImages/{category}/{data["trg_imname"]}'
        # get image set with no repetitions
        if source_fn not in files_no_repetition:
            files_no_repetition.append(source_fn)
        if target_fn not in files_no_repetition:
            files_no_repetition.append(target_fn)
        pairs_set_indices.append(files_no_repetition.index(source_fn))
        pairs_set_indices.append(files_no_repetition.index(target_fn))

        source_bbox = np.asarray(data["src_bndbox"])
        target_bbox = np.asarray(data["trg_bndbox"])
        # The source thresholds aren't actually used to evaluate PCK on SPair-71K, but for completeness
        # they are computed as well:
        thresholds.append(max(source_bbox[3] - source_bbox[1], source_bbox[2] - source_bbox[0]))
        thresholds.append(max(target_bbox[3] - target_bbox[1], target_bbox[2] - target_bbox[0]))

        source_size = data["src_imsize"][:2]  # (W, H)
        target_size = data["trg_imsize"][:2]  # (W, H)

        kp_ixs = torch.tensor([int(id) for id in data["kps_ids"]]).view(-1, 1).repeat(1, 3)
        source_raw_kps = torch.cat([torch.tensor(data["src_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        source_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=source_raw_kps)
        source_kps, src_x, src_y, src_scale = preprocess_kps_pad(source_kps, source_size[0], source_size[1], size)
        target_raw_kps = torch.cat([torch.tensor(data["trg_kps"], dtype=torch.float), torch.ones(kp_ixs.size(0), 1)], 1)
        target_kps = blank_kps.scatter(dim=0, index=kp_ixs, src=target_raw_kps)
        target_kps, trg_x, trg_y, trg_scale = preprocess_kps_pad(target_kps, target_size[0], target_size[1], size)

        kps.append(source_kps)
        kps.append(target_kps)
        inverse.append([src_x, src_y, src_scale])
        inverse.append([trg_x, trg_y, trg_scale])
    kps = torch.stack(kps)
    used_kps, = torch.where(kps[:, :, 2].any(dim=0))
    kps = kps[:, used_kps, :]
    print(f'Final number of used key points: {kps.size(1)}')
    num_imgs = len(thresholds)  # Total number of images (= 2 * number of pairs)
    torch.save(torch.arange(num_imgs).view(num_imgs // 2, 2), f'{output_path_pck}/pairs.pt')
    torch.save(torch.tensor(pairs_set_indices).view(num_imgs // 2, 2), f'{output_path_pck}/pairs_indices_in_set.pt')
    torch.save(torch.tensor(thresholds, dtype=torch.float), f'{output_path_pck}/pck_thresholds.pt')
    torch.save(torch.tensor(inverse), f'{output_path_pck}/inverse_coordinates.pt')
    torch.save(kps, f'{output_path_pck}/keypoints.pt')
    torch.save(SPAIR_PERMUTATIONS[category], f'{output_path_pck}/permutation.pt')
    return files_no_repetition, [None] * len(files_no_repetition)  # No bounding boxes are used


def load_spair_data_and_process(path, method, size, out_path, category='cat', split='test', image_format='png'):
    files_no_repetition, bboxes = load_spair_pck_data(path, category, split, size, out_path)
    process_images_and_save(out_path, files_no_repetition, method, size, bboxes, image_format)


def prepare_data(path, method, size, out, image_format, cub_acsm, spair_category, spair_split):
    if cub_acsm:  # Load CUB using ACSM pre-processing (this is the only dataset that uses bboxes in pre-processing)
        load_acsm_data_and_process(path, method=method, size=size, out_path=out, image_format=image_format)
    elif spair_category is not None:  # Load SPair-71K (bboxes = None)
        load_spair_data_and_process(path, method=method, size=size, out_path=out, category=spair_category,
                                    split=spair_split, image_format=image_format)
    else:  # Load images from a folder; bboxes = None
        load_image_folder_and_process(path, method=method, size=size, out_path=out, image_format=image_format)


if __name__ == "__main__":
    # Code taken from GANgealing (https://github.com/wpeebles/gangealing/blob/1f64782a35e0b623501c3e87f05240ca1cd82af6/prepare_data.py)
    # Same preprocessing of SPair-71K and CUB as in GANgealing, modified to match Neural Congealing files convention
    parser = argparse.ArgumentParser(description='Prepare data of benchmarks SPair-71K/CUB_200_2011.')
    parser.add_argument("--out", type=str, help="output path")
    parser.add_argument("--method", choices=['border', 'center', 'none'], default='center',
                        help='Algorithm to pad or crop input images to square. '
                             'border = border padding, center = center crop, none = no pre-processing')
    parser.add_argument("--path", type=str, default=None, help="path to image dataset (SPair-71K / CUB) or image set")
    parser.add_argument("--out_format", type=str, choices=['png', 'jpg'], default='png', help="format to store images")
    parser.add_argument("--size", type=int, default=256, help="resolution of images for the dataset")

    # Special arguments for loading SPair-71K and CUB for PCK evaluation purposes. If you use these options below,
    # you can ignore --path, --method and --out above.
    parser.add_argument("--spair_category", default=None, type=str, choices=list(SPAIR_PERMUTATIONS.keys()),
                        help='If specified, constructs the SPair-71K dataset for the specified category')
    parser.add_argument("--spair_split", default='test', choices=['trn', 'val', 'test'], type=str,
                        help='The split of SPair that will be constructed (only used if --spair_category is specified)')
    parser.add_argument("--cub_acsm", action='store_true',
                        help='If true, constructs the CUB dataset. This will use the same pre-processing and filtering '
                             'as the CUB validation split from GANgealing (and originally, ACSM paper).')

    args = parser.parse_args()

    Path('data').mkdir(parents=True, exist_ok=True)
    if args.cub_acsm:  # CUB_200_2011
        args.method = 'cub_crop'
        args.out = f'data/cub_subsets'
        download_cub_metadata('data')
    elif args.spair_category is not None:  # SPair-71K
        args.method = 'border'
        args.out = f'data/spair_sets/spair_{args.spair_category}_{args.spair_split}'

    if args.cub_acsm and args.path is None:  # Download CUB-200-2011 if needed
        args.path = download_cub('data')
    elif args.spair_category is not None and args.path is None:  # Download SPair-71K data if needed
        args.path = download_spair('data')
    else:
        assert args.path is not None

    prepare_data(args.path, args.method, args.size, args.out, args.out_format, args.cub_acsm, args.spair_category, args.spair_split)

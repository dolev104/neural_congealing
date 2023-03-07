from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils_atlas.utils_atlas_base import extract_upsampled_keys, extract_dino_saliency_map
from models.vit_extractor_model import get_feat_extractor


class AtlasDataset(Dataset):
    def __init__(self, config, device, dino_extractor=None):
        self.config = config
        self.device = device
        self.current_level = 0
        data_folder = Path(config["data_folder"])
        self.main_folder = data_folder
        self.image_set_name = data_folder.name

        if dino_extractor is not None:
            self.extractor = dino_extractor
        else:
            self.extractor = get_feat_extractor(config, device)
        self.dino_emb_size = self.extractor.embedding_dim
        self.atlas_resolution = config["atlas_resolution"]

        images_folder_path = data_folder.joinpath("images")
        saliency_masks_path = data_folder.joinpath("images_cosegmentation").joinpath("mask")
        input_files = sorted(
            list(images_folder_path.glob("*.jpeg")) + list(images_folder_path.glob("*.jpg")) + list(images_folder_path.glob("*.png")))  # input images
        self.input_files = input_files
        self.number_of_images = len(self.input_files)

        image_resolution = config["image_resolution"]
        resy, resx = image_resolution, image_resolution
        self.all_images = torch.zeros(
            (self.number_of_images, 3, resy, resx), requires_grad=False
        )
        if self.atlas_resolution == 0:  # relevant only for inspect_similarity_set, not for training/eval
            feat_resy = int(self.extractor.p / self.extractor.stride[0] * (resy // self.extractor.p - 1) + 1)
            feat_resx = int(self.extractor.p / self.extractor.stride[1] * (resx // self.extractor.p - 1) + 1)
        else:
            feat_resy, feat_resx = self.atlas_resolution, self.atlas_resolution
        self.imgs_dino_keys = torch.zeros(
            (self.number_of_images, self.dino_emb_size, feat_resy, feat_resx),
            requires_grad=False,
        )
        self.imgs_saliency_maps = torch.zeros(
            (self.number_of_images, 1, feat_resy, feat_resx), requires_grad=False
        )
        self.images_state = torch.zeros(self.number_of_images, dtype=torch.int8)  # 0 - regular image, 1 - horizontally flipped
        if config["ext_horizontal_flips"]:
            self.all_images_hor_flipped = torch.zeros(
                (self.number_of_images, 3, resy, resx), requires_grad=False
            )
            self.imgs_dino_keys_hor_flipped = torch.zeros(
                (self.number_of_images, self.dino_emb_size, feat_resy, feat_resx),
                requires_grad=False,
            )
            if self.config["use_masks"]:
                self.imgs_saliency_maps_hor_flipped = torch.zeros(
                    (self.number_of_images, 1, feat_resy, feat_resx), requires_grad=False
                )

        self.load_data(saliency_masks_path)

        self.init_atlas_dict = {
            "all_saliency_masks": self.imgs_saliency_maps,
            "all_dino_keys": self.imgs_dino_keys
        }

    def load_data(self, saliency_masks_path):
        print("Uploading image set data...")
        for i in tqdm(range(self.number_of_images)):
            file1 = self.input_files[i]
            im_pil = Image.open(str(file1)).convert("RGB")
            im = transforms.ToTensor()(im_pil)

            saliency_from_path = f"{saliency_masks_path}/{Path(file1).stem}_mask.png"

            im_keys, saliency_map = self.extract_dino_features(im_pil, saliency_from_path=saliency_from_path)
            self.all_images[i] = im
            self.imgs_dino_keys[i] = im_keys
            if self.config["use_masks"]:
                self.imgs_saliency_maps[i] = saliency_map  # [1, resy, resx]

            if self.config["ext_horizontal_flips"]:
                im_pil_hflip = im_pil.transpose(Image.FLIP_LEFT_RIGHT)
                self.all_images_hor_flipped[i] = transforms.ToTensor()(im_pil_hflip)
                if self.config["use_masks"]:
                    self.imgs_saliency_maps_hor_flipped[i] = transforms.functional.hflip(saliency_map)

                im_keys_flipped, _ = self.extract_dino_features(im_pil_hflip, extract_saliency=False)
                self.imgs_dino_keys_hor_flipped[i] = im_keys_flipped

    def extract_dino_features(self, input_image, saliency_from_path=None, extract_saliency=True):
        if isinstance(input_image, torch.Tensor):
            h, w = input_image.shape[-2:]
            prep_img = self.extractor.normalize_image(input_image)
        else:  # PIL image
            w, h = input_image.size[-2:]
            prep_img = self.extractor.preprocess_pil_image(input_image)

        with torch.no_grad():
            processed_keys = extract_upsampled_keys(prep_img, self.extractor, feat_res=self.atlas_resolution,
                                dino_keys_layer=self.config["dino_keys_layer"], facet="key",
                                preprocess_image=False).detach()[0].cpu()  # [emb, h, w]
        if extract_saliency and self.config["use_masks"]:
            saliency_map = extract_dino_saliency_map(saliency_from_path, self.extractor, (h, w),
                                                     feat_res=self.atlas_resolution).cpu()  # [1, h, w]
        else:
            saliency_map = None
        return processed_keys, saliency_map

    def get_img_data(self, im_idx):
        curr_im = self.all_images[im_idx]
        dino_keys = self.imgs_dino_keys[im_idx]
        saliency_map = self.imgs_saliency_maps[im_idx]
        return curr_im, dino_keys, saliency_map

    def get_hor_flipped_img_data(self, im_idx):
        curr_im = self.all_images_hor_flipped[im_idx]
        dino_keys = self.imgs_dino_keys_hor_flipped[im_idx]
        saliency_map = self.imgs_saliency_maps_hor_flipped[im_idx]
        return curr_im, dino_keys, saliency_map

    def __getitem__(self, index):
        if self.config["ext_horizontal_flips"]:
            input_image, dino_keys, saliency_map, input_image_fp, dino_keys_fp, saliency_map_fp = self.get_sample_data(index)
        else:
            input_image, dino_keys, saliency_map = self.get_sample_data(index)
            input_image_fp = None

        sample = {"input_image": input_image, "input_keys": dino_keys,
                  "input_saliency": saliency_map, "current_im_idx": index}
        if input_image_fp is not None:
            sample["input_image_fp"] = input_image_fp
            sample["input_keys_fp"] = dino_keys_fp
            sample["input_saliency_fp"] = saliency_map_fp

        return sample

    def get_sample_data(self, index, allow_return_all=True):
        original_data_tuple = self.get_img_data(index)  # (image, dino keys, saliency map)
        if self.config["ext_horizontal_flips"]:
            flipped_data_tuple = self.get_hor_flipped_img_data(index)  # flipped -- (image, dino keys, saliency map)

            ret = original_data_tuple + flipped_data_tuple
            if self.images_state[index] > 0:
                ret = flipped_data_tuple + original_data_tuple  # flipped data is used to update the joint atlas
            if not allow_return_all:
                return ret[:len(original_data_tuple)]
        else:
            ret = original_data_tuple
        return ret

    def get_data_for_eval_training(self):
        return self.all_images, self.imgs_dino_keys, self.all_images_hor_flipped, self.imgs_dino_keys_hor_flipped

    def __len__(self):
        return self.number_of_images

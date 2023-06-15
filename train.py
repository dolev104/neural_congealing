import random
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import AtlasDataset
from utils_atlas.utils_atlas_base import upload_checkpoint_for_training, load_state_opt_criterion
from losses import AtlasLoss
from utils_atlas.logger import DataLogger

from models.model import AtlasModel

device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")


def train_model(config):
    if config["upload_trained_model_checkpoint"]:
        checkpoint, config, dataset, model = upload_checkpoint_for_training(config, AtlasModel, AtlasDataset, device)
    else:
        dataset = AtlasDataset(config, device)
        model = AtlasModel(config, dataset.dino_emb_size, dataset.init_atlas_dict, device).to(device)
    data_loader = DataLoader(dataset, config["batch_size"], shuffle=True, pin_memory=True)

    criterion = AtlasLoss(config, dataset.dino_emb_size, model.atlas_size, dataset.init_atlas_dict, device)

    optimizer_stns = torch.optim.Adam(model.stn.parameters(), lr=config["learning_rate_stn"])
    optimizer_atlas = torch.optim.Adam([model.atlas_keys, model.atlas_saliency], lr=config["learning_rate_atlas"])

    logger = DataLogger(config, dataset, model, device, log_to_wandb=config["log_to_wandb"])

    if config["upload_trained_model_checkpoint"]:
        criterion, optimizer_atlas, optimizer_stns = load_state_opt_criterion(config, checkpoint, criterion, optimizer_atlas, optimizer_stns)

    train(config, data_loader, dataset, model, criterion, optimizer_stns, optimizer_atlas, logger)


def train(config, data_loader, dataset, model, criterion, optimizer_stns, optimizer_atlas, logger):
    n_iters = config["n_iters"]
    print("running with device: ", device)

    if config["ext_horizontal_flips"] or config["ext_gradual_atlas_training"]:
        ext_update_imgs_every = config["ext_update_flips_and_imgs_to_update_atlas_every"]
    else:
        ext_update_imgs_every = None

    with tqdm(range(n_iters)) as tepoch:
        for epoch in tepoch:
            if config["bootstrap_stn_sim"] >= 0 and epoch == config["bootstrap_stn_sim"]:  # time to bring in STNflow
                model.add_stn_flow_to_training()

            for i, input_load in enumerate(data_loader):
                inputs = {
                    "current_im_idx": input_load["current_im_idx"],
                    "input_image": input_load["input_image"].to(device),
                    "input_keys": input_load["input_keys"].to(device),
                    "input_saliency": input_load["input_saliency"].to(device),
                }

                outputs = model(inputs)
                total_loss, losses = criterion(outputs, inputs, sim_only_bootstrap=(not model.with_flow))

                if config["ext_horizontal_flips"]:
                    new_losses, total_loss_all = mapping_fw_pass(input_load, model, criterion)
                    total_loss += total_loss_all
                    losses.update(new_losses)

                optimizer_atlas.zero_grad()
                optimizer_stns.zero_grad()
                total_loss.backward()
                optimizer_atlas.step()
                optimizer_stns.step()

                if (config["ext_horizontal_flips"] or config["ext_gradual_atlas_training"]) and i == 0 and epoch % ext_update_imgs_every == 0 and epoch > 0:
                    update_modules(criterion, model, dataset, config)

                log_data = logger.log_data(epoch, i, losses, model, dataset, inputs, optimizer_stns, optimizer_atlas, ext_update_imgs_every, criterion.imgs_idx_to_update_atlas,
                                           criterion.all_imgs_update_atlas, sim_only_bootstrap=(not model.with_flow), log_to_wandb=config["log_to_wandb"])

                if config["log_to_wandb"]:
                    wandb.log(log_data)
                else:
                    logger.save_locally(log_data, i, epoch)
                plt.close('all')  # close all open figures

            tepoch.set_description(f"Epoch {epoch}")
            tepoch.set_postfix(loss=total_loss.item())


def mapping_fw_pass(input_load, model, criterion):
    inputs_fp = {
        "current_im_idx": input_load["current_im_idx"],
        "input_image": input_load["input_image_fp"].to(device),
        "input_keys": input_load["input_keys_fp"].to(device),
        "input_saliency": input_load["input_saliency_fp"].to(device),
    }
    outputs_fp = model(inputs_fp)
    total_loss_fp, losses_fp = criterion(outputs_fp, inputs_fp, sim_only_bootstrap=(not model.with_flow), mapping_losses_only=True)
    new_losses = {}
    for k, v in losses_fp.items():
        if "loss" in k:
            new_losses["fp_" + k] = v
    return new_losses, total_loss_fp


def update_modules(criterion, model, dataset, config):
    with torch.no_grad():
        atlas_keys, atlas_saliency = model.get_atlas_params()
        atlas_keys, atlas_saliency = atlas_keys[None, ...], atlas_saliency[None, ...]  # [1, 384, res_atlas, res_atlas], [1, res_atlas, res_atlas]

        all_images, imgs_dino_keys, all_images_hor_flipped, imgs_dino_keys_hor_flipped = dataset.get_data_for_eval_training()
        all_images, imgs_dino_keys = all_images.to(device), imgs_dino_keys.to(device)

        _, orig_imgs_keys, orig_white_mask = model.process_batch(all_images, imgs_dino_keys,
                                                                 padding_mode_im=model.config["stn_padding_mode_im"], padding_mode_keys=model.config["stn_padding_mode_keys"],
                                                                 return_congealed_keys=True, return_oob_mask=True)
        saliency_for_keys_loss = atlas_saliency * orig_white_mask.squeeze(1)  # batch, 128, 128 ---- 1128 change 2240
        keys_mse_loss, keys_cosine_loss = criterion.get_total_keys_loss(atlas_keys, orig_imgs_keys, saliency_for_keys_loss, config["keys_output_loss_coeff"], config["keys_output_cosine_loss_coeff"], dims_reduce_all=False)  # [number_of_images]
        total_loss_orig_keys = keys_mse_loss + keys_cosine_loss

        if config["ext_horizontal_flips"]:
            all_images_hor_flipped, imgs_dino_keys_hor_flipped = all_images_hor_flipped.to(device), imgs_dino_keys_hor_flipped.to(device)
            _, flipped_hor_imgs_keys, flipped_hor_white_mask = model.process_batch(all_images_hor_flipped, imgs_dino_keys_hor_flipped,
                                                                     padding_mode_im=model.config["stn_padding_mode_im"], padding_mode_keys=model.config["stn_padding_mode_keys"],
                                                                     return_congealed_keys=True, return_oob_mask=True)

            saliency_for_keys_loss = atlas_saliency * flipped_hor_white_mask.squeeze(1)
            keys_mse_loss_f, keys_cosine_loss_f = criterion.get_total_keys_loss(atlas_keys, flipped_hor_imgs_keys, saliency_for_keys_loss, config["keys_output_loss_coeff"], config["keys_output_cosine_loss_coeff"], dims_reduce_all=False)  # [number_of_images]
            total_loss_flipped_hor_keys = keys_mse_loss_f + keys_cosine_loss_f

            stacked_losses = torch.stack((total_loss_orig_keys, total_loss_flipped_hor_keys), dim=0)  # [2, number_of_images]
        else:
            stacked_losses = total_loss_orig_keys  # [number_of_images]

        # update dataset module: which images should be flipped
        if config["ext_horizontal_flips"]:
            min_vals = torch.min(stacked_losses, dim=0).indices  # number_of_images
            dataset.images_state = min_vals  # 0 - original, 1 - horizontal flip

        if config["ext_gradual_atlas_training"]:
            # update losses module: add an additional image to be used for updating the atlas from the ones not in imgs_idx_to_update_atlas
            if criterion.imgs_idx_to_update_atlas.long().sum() < dataset.number_of_images:
                if config["ext_horizontal_flips"]:
                    new_indices = dataset.images_state.long()
                    stacked_losses_updated = torch.stack([stacked_losses[new_indices[i], i] for i in range(dataset.number_of_images)])
                else:
                    stacked_losses_updated = stacked_losses
                stacked_losses_updated_tmp = stacked_losses_updated.clone()
                for i in range(dataset.number_of_images):
                    if criterion.imgs_idx_to_update_atlas[i]:
                        stacked_losses_updated_tmp[i] = stacked_losses_updated.max() + 1
                    else:
                        stacked_losses_updated_tmp[i] = stacked_losses_updated[i]
                new_image_idx = stacked_losses_updated_tmp.argmin()

                criterion.imgs_idx_to_update_atlas[new_image_idx] = True
            else:
                criterion.all_imgs_update_atlas = True


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        default="./configs/config.yaml",
        help="Config path",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # set seed
    seed = config["seed"]
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config["seed"] = seed

    image_set_name = str(Path(config["data_folder"]).name)

    if config["log_to_wandb"]:
        import wandb

        wandb.init(project=config["wandb_project"], entity=config["wandb_entity"], config=config, name=image_set_name)
        wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
        config = dict(wandb.config)
    else:
        now = datetime.now()
        run_name = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_{image_set_name}"
        path = Path(f"{config['results_folder']}/{run_name}")
        path.mkdir(parents=True, exist_ok=True)
        config["results_folder"] = str(path)
        with open(path / "config.yaml", "w") as f:
            yaml.dump(config, f)

    train_model(config)
    if config["log_to_wandb"]:
        wandb.finish()

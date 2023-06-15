import math

import colorsys
from pathlib import Path

import cv2
import numpy as np
import torch
import wandb
from PIL import Image
from matplotlib import pyplot as plt, cm
from sklearn.decomposition import PCA
from torchvision import transforms
from torchvision.io import write_video
from torchvision.utils import save_image
from tqdm import tqdm

from models.spatial_transformers.spatial_transformer import SpatialTransformer
from .utils_atlas_base import tensor2im, tensor2numpy, get_keys_loss_func, apply_pca, calculate_fw_matrix, \
    plot_images_grid, apply_griddata
from .vis_tools.helpers import splat_points


def get_trained_pca(dataset, pca_n_components=4, pca_visualize_with_global_minmax=False, log_to_wandb=True, eval=False):
    all_keys = dataset.imgs_dino_keys
    all_saliency_maps = dataset.imgs_saliency_maps
    masked_keys = ((all_saliency_maps >= 0.5) * all_keys).permute(0, 2, 3, 1)
    masked_keys = masked_keys[masked_keys.abs().sum(dim=-1) > 0]
    trained_pca = PCA(n_components=pca_n_components).fit(masked_keys)

    # get grid of image set
    n, c, h, w = dataset.imgs_dino_keys.shape
    pca_descriptors = apply_pca(dataset.imgs_dino_keys.permute(0, 2, 3, 1).reshape(-1, c), pca_n_components, trained_pca=trained_pca)
    if pca_visualize_with_global_minmax:
        min_vals, max_vals = pca_descriptors.min(), pca_descriptors.max()
    else:
        min_vals, max_vals = pca_descriptors.min(dim=0).values, pca_descriptors.max(dim=0).values
        min_vals, max_vals = min_vals[None, ...], max_vals[None, ...]
    pca_data_keys_scaled = (pca_descriptors - min_vals) / (max_vals - min_vals)
    pca_data_keys_scaled = pca_data_keys_scaled.reshape((n, h, w, pca_n_components)).permute(0, 3, 1, 2)  # [batch, num_comp, h, w]

    virdis_colormap = cm.get_cmap('viridis')
    num_images = pca_data_keys_scaled.shape[0]
    imgs_pca_logging_dict = {}
    nrow = int(math.ceil(math.sqrt(num_images))) + 1  # assumes num_images > 2

    for i in range(pca_n_components):
        keys_pca_i = []
        for im_idx in range(num_images):
            keys_pca_i.append(virdis_colormap(pca_data_keys_scaled[im_idx, i].cpu()))
        keys_pca_i = torch.from_numpy(np.stack(keys_pca_i, axis=0)).permute(0, 3, 1, 2)  # [batch, c, h, w]
        keys_pca_i_grid = plot_images_grid(keys_pca_i, nrow=nrow, normalize=False)
        imgs_pca_logging_dict[f"pca_vis_orig_imgs_keys/pca_ch{i + 1}"] = wandb.Image(
            tensor2im(keys_pca_i_grid)) if log_to_wandb else keys_pca_i_grid

    if pca_n_components > 3:
        keys_pca234 = pca_data_keys_scaled[:, 1:4].cpu()
        keys_pca234_grid = plot_images_grid(keys_pca234, nrow=nrow, normalize=False)
        imgs_pca_logging_dict["pca_vis_orig_imgs_keys/pca_ch234"] = wandb.Image(tensor2im(keys_pca234_grid)) \
            if log_to_wandb else keys_pca234_grid
        if not log_to_wandb and eval:  # also save whole tensor
            imgs_pca_logging_dict[f"pca_vis_orig_imgs_keys/pca_ch234_sep"] = keys_pca234

    return trained_pca, min_vals[None, ...], max_vals[None, ...], imgs_pca_logging_dict


def get_pca_visualizations(atlas_keys, pca_n_components, trained_pca=None, pca_min_vals=None, pca_max_vals=None,
                           return_pca_tensor=False, pca_global_minmax=False):
    keys_desc = atlas_keys[0].detach().cpu()
    c, resy, resx = keys_desc.shape
    pca_atlas_keys = apply_pca(keys_desc.reshape(c, -1).t(), pca_n_components, trained_pca)
    pca_atlas_keys = pca_atlas_keys.reshape((resy, resx, pca_n_components))

    if pca_min_vals is not None and pca_max_vals is not None:
        pca_atlas_keys_rescaled = ((pca_atlas_keys - pca_min_vals) / (pca_max_vals - pca_min_vals)).clip(min=0., max=1.)
    else:
        if pca_global_minmax:
            pca_min_vals = pca_atlas_keys.min()
            pca_max_vals = pca_atlas_keys.max()
        else:
            pca_min_vals = pca_atlas_keys.reshape(-1, pca_n_components).min(dim=0).values
            pca_max_vals = pca_atlas_keys.reshape(-1, pca_n_components).max(dim=0).values
        pca_atlas_keys_rescaled = ((pca_atlas_keys - pca_min_vals) / (pca_max_vals - pca_min_vals)).clip(min=0., max=1.)

    if return_pca_tensor:
        return pca_atlas_keys_rescaled
    else:  # plot fig
        num_imgs = pca_n_components + (pca_n_components > 3)
        num_cols = int(math.ceil(math.sqrt(num_imgs)))
        num_rows = int(math.ceil(num_imgs / num_cols))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
        [axi.set_axis_off() for axi in axes.ravel()]
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig_fontsize = 8

        for i in range(pca_n_components):
            axes[i // num_cols, i % num_cols].imshow(pca_atlas_keys_rescaled[:, :, i])
            axes[i // num_cols, i % num_cols].set_title(f"PCA channel {i + 1}", fontsize=fig_fontsize)
        if pca_n_components > 3:
            axes[pca_n_components // num_cols, pca_n_components % num_cols].imshow(pca_atlas_keys_rescaled[:, :, 1:4])
            axes[pca_n_components // num_cols, pca_n_components % num_cols].set_title(f"PCA channels 234", fontsize=fig_fontsize)

        return fig


class DataLogger:
    def __init__(self, config, dataset, model, device, log_to_wandb=True, eval=False):
        self.config = config
        self.device = device
        self.pca_n_components = config["pca_n_components"]
        self.jet_colormap = plt.get_cmap('jet')
        self.save_imgs_of_atlas = True  # relevant only for gradual atlas training

        self.keys_loss_func = get_keys_loss_func()
        self.mse_loss = torch.nn.MSELoss()

        self.pca_global_minmax = self.config["pca_visualize_with_global_minmax"]
        # apply PCA on datasets' keys, to get the principal components of the data
        self.trained_pca, self.pca_min_vals, self.pca_max_vals, imgs_pca_logging_dict = get_trained_pca(dataset, pca_n_components=self.pca_n_components, pca_visualize_with_global_minmax=self.pca_global_minmax, log_to_wandb=log_to_wandb, eval=eval)
        if log_to_wandb:
            wandb.log(imgs_pca_logging_dict)
        else:
            self.imgs_pca_logging_dict = imgs_pca_logging_dict
            if not eval:
                imgs_pca_logging_dict["epoch"] = 'pca_vis_input_set'
                self.save_locally(imgs_pca_logging_dict, 0, 0)

        if self.config["save_model_initialization"] and log_to_wandb:
            filename = f"model_checkpoint_init.pt"
            dict_to_save = {'model': model.state_dict()}
            if log_to_wandb:
                checkpoint_path = f"{wandb.run.dir}/{filename}"
            else:
                checkpoint_path = f"{self.config['results_folder']}/{filename}"
            torch.save(dict_to_save, checkpoint_path)

        if self.config["use_griddata"] or eval:
            image_resolution = self.config["image_resolution"]
            resy, resx = image_resolution, image_resolution
            self.out_indices = torch.cartesian_prod(torch.arange(resy), torch.arange(resx)).float().numpy()

    @torch.no_grad()
    def log_data(self, epoch, curr_batch_idx, losses, model, dataset, inputs, optimizer_stns, optimizer_atlas, ext_update_imgs_every, imgs_idx_to_update_atlas,
                 all_imgs_update_atlas, sim_only_bootstrap=False, log_to_wandb=True):
        log_data = {}

        log_data["epoch"] = epoch
        if curr_batch_idx <= 0 and epoch % self.config["log_losses_freq"] == 0 and log_to_wandb:
            for loss_key, loss_val in losses.items():
                if "loss" in loss_key:
                    log_data[f"Loss/{loss_key}"] = loss_val
                elif "transformation_params" == loss_key:
                    theta, s, t_x, t_y = loss_val.t()  # theta, s, tx, ty each of shape [batch_size]
                    for i in range(len(inputs['current_im_idx'])):
                        curr_im_idx = inputs['current_im_idx'][i].item()
                        log_data[f"current_im_idx_batchidx{i}"] = curr_im_idx
                        log_data[f"Transformations_theta/im_{curr_im_idx}_theta_deg"] = (theta[i] * 180 / math.pi).item()
                        log_data[f"Transformations_scale/im_{curr_im_idx}_scale"] = s[i].item()
                        log_data[f"Transformations_tx/im_{curr_im_idx}_tx"] = t_x[i].item()
                        log_data[f"Transformations_ty/im_{curr_im_idx}_ty"] = t_y[i].item()

        if ext_update_imgs_every is not None and curr_batch_idx == 0 and epoch % ext_update_imgs_every == 0:
            if self.config["ext_horizontal_flips"] and log_to_wandb:
                for i in range(len(dataset.images_state)):
                    log_data["images_state/img_" + str(i)] = dataset.images_state[i].item()

            # log images that participate in atlas update: imgs_idx_to_update_atlas
            if self.config["ext_gradual_atlas_training"] and self.save_imgs_of_atlas:
                new_indices = dataset.images_state.long()
                if self.config["ext_horizontal_flips"]:
                    stacked_imgs = torch.stack((dataset.all_images, dataset.all_images_hor_flipped), dim=0)  # [2, num_images, 3, resy, resx]
                else:
                    stacked_imgs = dataset.all_images[None, ...]  # [1, num_images, 3, resy, resx]
                current_images_to_update_atlas = torch.stack([stacked_imgs[new_indices[i], i] for i in range(dataset.number_of_images) if imgs_idx_to_update_atlas[i]], dim=0)
                img_updating_atlas = plot_images_grid(current_images_to_update_atlas, nrow=int(math.ceil(math.sqrt(current_images_to_update_atlas.shape[0]))), normalize=False)
                log_data.update({f"Images_to_update_atlas/Images_updating_atlas": wandb.Image(tensor2im(img_updating_atlas)) if log_to_wandb else img_updating_atlas})
                if imgs_idx_to_update_atlas.long().sum() == dataset.all_images.shape[0]:
                    self.save_imgs_of_atlas = False

        # log images and visualizations
        if curr_batch_idx <= 0 and epoch % self.config["log_images_freq"] == 0:
            model.eval()
            if epoch % self.config["log_additional_vis_freq"] == 0 and epoch > 0:
                reduced_logging = False
            else:
                reduced_logging = True
            evaluation_data = self.evaluate_and_log(dataset, model, sim_only_bootstrap=sim_only_bootstrap,
                                                    reduced_logging=reduced_logging, log_to_wandb=log_to_wandb)
            log_data.update(evaluation_data)
            model.train()

        # save model checkpoint
        if curr_batch_idx <= 0 and epoch >= self.config["save_model_starting_epoch"] and epoch % self.config["save_model_freq"] == 0:
            filename = f"checkpoint_epoch_{epoch}.pt"
            dict_to_save = {
                'epoch': epoch,
                'config': self.config,
                'model': model.state_dict(),
                'images_state': dataset.images_state.detach().cpu(),
            }
            if self.config["save_complete_state"]:  # in case of wanting to continue training
                dict_to_save['stn_optimizer_state_dict'] = optimizer_stns.state_dict()
                dict_to_save['atlas_optimizer_state_dict'] = optimizer_atlas.state_dict()
                if self.config["ext_gradual_atlas_training"]:
                    dict_to_save['imgs_idx_to_update_atlas'] = imgs_idx_to_update_atlas.detach().cpu() \
                        if imgs_idx_to_update_atlas is not None else imgs_idx_to_update_atlas
                    dict_to_save['all_imgs_update_atlas'] = all_imgs_update_atlas
            if log_to_wandb:
                checkpoint_path = f"{wandb.run.dir}/{filename}"
            else:
                checkpoint_path = f"{self.config['results_folder']}/{filename}"
            torch.save(dict_to_save, checkpoint_path)
        return log_data

    @staticmethod
    def get_colored_letters_atlas(atlas_size):
        # based on code from https://github.com/ykasten/layered-neural-atlases/blob/19aa32dd0cf0de7e92d279fea82844f28a15d4a0/evaluate.py#L87
        """ Create a text pattern in order to visualize the mapping functions.
            NOTE: the hardcoded numbers fit to a specific atlas resolution (128 x 128). """
        letters_image = cv2.UMat(tensor2numpy(torch.zeros(atlas_size)))
        start, end, jump = 20, 200, 40
        fontscale_small, fontscale_cap = 0.9, 0.8
        starting_pnt_y = 1
        font_thickness = 2
        loc_tensor = range(start, end, jump)
        colors = ((torch.tensor(loc_tensor) - start) / end).numpy()
        colors_num = len(colors)
        for i, ii in enumerate(loc_tensor):
            start_color = colors[i]
            cur_color = colorsys.hsv_to_rgb(start_color, 1.0, 1.0)
            cv2.putText(letters_image,
                        'abcdefghijlmnopqrstuvwxyz1234567890!@#$%^&*()-+=>', (starting_pnt_y, ii - start // 2 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale_small, cur_color, font_thickness, cv2.LINE_AA)
            start_color = colors[(i + 3) % colors_num]
            cur_color = colorsys.hsv_to_rgb(start_color, 1.0, 1.0)
            cv2.putText(letters_image,
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ?~;:<./\|][{},', (starting_pnt_y, ii + start // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fontscale_cap, cur_color, font_thickness, cv2.LINE_AA)

        return transforms.ToTensor()(cv2.UMat.get(letters_image)).unsqueeze(0)

    @torch.no_grad()
    def evaluate_and_log(self, dataset, model, log_to_wandb=True, sim_only_bootstrap=False, reduced_logging=True, eval_only_mode=False):
        evaluation_data = {}

        all_curr_images = []
        all_congealed_imgs_sim = []
        all_congealed_imgs = []
        all_congealed_imgs_checkerboard = []
        all_affine_params = []
        propagated_edits_flow = []

        image_resolution = self.config["image_resolution"]
        im_resy, im_resx = image_resolution, image_resolution
        checkerboard_image = transforms.ToTensor()(Image.open('data/checkerboard_gray_256.png').convert("RGB"))[None, ...].to(self.device)

        if eval_only_mode:
            all_congealed_saliencies = []
            all_sampling_grids = []
            evaluation_data["checkerboard_image"] = checkerboard_image

        atlas_keys, atlas_saliency = model.get_atlas_params()  # saliency of shape [h, w]
        atlas_keys = atlas_keys[None, ...]  # [1, emb, h, w]
        resized_atlas_saliency = transforms.Resize(im_resx)(atlas_saliency[None, None, ...])
        atlas_resy, atlas_resx = atlas_keys.shape[-2:]
        # atlas visualizations
        evaluation_data["atlas_vis/atlas_keys_pca_of_set"] = get_pca_visualizations(atlas_keys, self.pca_n_components, trained_pca=self.trained_pca, pca_min_vals=self.pca_min_vals, pca_max_vals=self.pca_max_vals, return_pca_tensor=eval_only_mode)
        evaluation_data["atlas_vis/atlas_saliency"] = wandb.Image(tensor2im(atlas_saliency[None, None, ...])) if log_to_wandb else atlas_saliency
        if eval_only_mode:
            evaluation_data["atlas_vis/atlas_keys_pca"] = get_pca_visualizations(atlas_keys * (atlas_saliency > 0.5), self.pca_n_components, return_pca_tensor=True, pca_global_minmax=self.pca_global_minmax)
            evaluation_data["atlas_vis/resized_atlas_saliency"] = resized_atlas_saliency

        # get text in atlas space
        letters_rgb_atlas_space = self.get_colored_letters_atlas(torch.Size((1, 3, atlas_resy, atlas_resx))).to(self.device)
        letters_alpha_atlas_space = torch.where(letters_rgb_atlas_space.sum(dim=1) == 0., 0., 1.)
        total_alpha_latters = letters_alpha_atlas_space * atlas_saliency[None, ...]
        letters_image_atlas_space = torch.cat((letters_rgb_atlas_space, total_alpha_latters.unsqueeze(0)), dim=1)
        if not sim_only_bootstrap:  # for visualizing mappings
            threshold = -1 if self.config["use_griddata"] else 0  # griddata requires all points
            i, j = torch.where(total_alpha_latters.squeeze() > threshold)  # i indexes height, j indexes width. check where alpha > 0
            points = torch.stack([j, i], -1).unsqueeze(0).to(self.device)  # (1, P, 2); points are stored in (x, y) format
            alpha_channel_atlas_letters = letters_image_atlas_space[:, 3:4, i, j].permute(0, 2, 1)  # (1, P, 1), [0, 1]
            colors_atlas_letters = letters_image_atlas_space[:, :3, i, j].permute(0, 2, 1)  # (1, P, 3), [0, 1]
            points_normalized_atlas_letters = SpatialTransformer.normalize(points, atlas_resy, atlas_resy)

        for im_idx in tqdm(range(dataset.number_of_images)):
            input_image, input_keys, input_saliency = dataset.get_sample_data(im_idx, allow_return_all=False)
            input_image, input_keys, input_saliency = input_image[None, ...].to(self.device), input_keys[None, ...].to(self.device), input_saliency[None, ...].to(self.device)
            all_curr_images.append(input_image)

            ret = model.process_batch(input_image, input_keys=input_keys, input_saliency=input_saliency,
                                      im_output_res=input_image.shape[-1], padding_mode_im=self.config["stn_padding_mode_im"],
                                      padding_mode_keys=self.config["stn_padding_mode_keys"],
                                      padding_mode_saliency=self.config["stn_padding_mode_saliency"],
                                      return_congealed_keys=True, return_congealed_sal=True, return_images=True)
            transformation_params, congealed_keys, congealed_saliency, final_congealed_image, congealed_sim = ret
            bw_sim_mat, affine_params, delta_flow, final_sampling_grid = transformation_params
            all_affine_params.append(affine_params)
            if eval_only_mode:
                all_sampling_grids.append(bw_sim_mat if sim_only_bootstrap else final_sampling_grid)
                all_congealed_saliencies.append(congealed_saliency)

            congealed_image_chck = final_congealed_image * resized_atlas_saliency + (1 - resized_atlas_saliency) * checkerboard_image

            if sim_only_bootstrap:
                all_congealed_imgs_sim.append(final_congealed_image)
            else:
                all_congealed_imgs.append(final_congealed_image)
                all_congealed_imgs_checkerboard.append(congealed_image_chck)
                all_congealed_imgs_sim.append(congealed_sim)

                # get text on atlas image -- using final flow
                upoints = model.uncongeal_points(input_image, points_normalized_atlas_letters,
                                                 samp_grid=final_sampling_grid.detach(), normalize_input_points=False)  # [num images, pnts, 2]
                curr_colors_atlas_letters, curr_alpha_channel_atlas_letters = colors_atlas_letters, alpha_channel_atlas_letters
                if self.config["use_griddata"]:
                    letters_rgba = torch.cat((curr_colors_atlas_letters, curr_alpha_channel_atlas_letters), dim=-1).cpu().numpy()[0]
                    propagated_edit_img = apply_griddata(input_image.cpu(), upoints.cpu().numpy()[0], letters_rgba, self.out_indices)
                else:
                    propagated_edit_img = splat_points(input_image, upoints, sigma=1.0, opacity=0.9, colorscale='plasma',
                                                   colors=curr_colors_atlas_letters, alpha_channel=curr_alpha_channel_atlas_letters)
                propagated_edits_flow.append(propagated_edit_img)

            if not reduced_logging:
                keys_error_masked = ((congealed_saliency * (atlas_keys - congealed_keys)).norm(dim=1) ** 2)
                keys_error_masked = (keys_error_masked - keys_error_masked.min()) / (keys_error_masked.max() - keys_error_masked.min())
                saliency_error_im = self.jet_colormap(((atlas_saliency - congealed_saliency.squeeze()) ** 2).cpu().numpy())
                evaluation_data[f"keys_error_masked/keys_error_masked_im_{im_idx:03d}"] = [wandb.Image(tensor2im(keys_error_masked[None, ...]))] \
                    if log_to_wandb else keys_error_masked
                evaluation_data[f"saliency_error/saliency_error_im_{im_idx:03d}"] = [wandb.Image(saliency_error_im)] \
                    if log_to_wandb else transforms.ToTensor()(saliency_error_im)
                if delta_flow is not None:
                    assert delta_flow.shape[-1] == 2, f"delta_flow has shape {delta_flow.shape}"
                    delta_flow_n = ((delta_flow - delta_flow.min()) / (delta_flow.max() - delta_flow.min())).clamp(min=0., max=1.)
                    delta_flow_n = torch.cat((delta_flow_n.detach().cpu(), torch.zeros(*delta_flow.shape)[..., [0]]), dim=-1).permute(0, 3, 1, 2)
                    evaluation_data[f"Transformations_delta_flow_vis/delta_flow_im_{im_idx:03d}"] = [wandb.Image(tensor2im(delta_flow_n))] \
                        if log_to_wandb else delta_flow_n

        all_curr_images = torch.cat(all_curr_images, dim=0)
        all_congealed_imgs_sim = torch.cat(all_congealed_imgs_sim, dim=0)

        # get text on atlas image -- using STNsim only
        all_affine_params = torch.cat(all_affine_params, dim=0)  # [num images, 4]
        fw_matrix = calculate_fw_matrix(all_affine_params)
        letters_image_atlas_space = torch.cat((letters_rgb_atlas_space, letters_alpha_atlas_space.unsqueeze(0)), dim=1)
        text_edited_im_letters_only = model.apply_bw_warp(letters_image_atlas_space.expand(all_affine_params.shape[0], -1, -1, -1).to(self.device),
                                                          fw_matrix, flow=None, padding_mode='zeros', img_size=all_curr_images.size())
        letters_alpha = text_edited_im_letters_only[:, 3].unsqueeze(1)
        all_letters_on_orig = text_edited_im_letters_only[:, :3] * letters_alpha + all_curr_images * (1 - letters_alpha)

        if eval_only_mode:
            evaluation_data["all_input_images"] = all_curr_images
            evaluation_data["all_unmasked_congealed_imgs_sim"] = all_congealed_imgs_sim
            evaluation_data["all_congealed_saliencies"] = torch.cat(all_congealed_saliencies, dim=0)
            evaluation_data["all_sampling_grids"] = torch.cat(all_sampling_grids, dim=0)

        if not sim_only_bootstrap:
            propagated_edits_flow = torch.cat(propagated_edits_flow, dim=0)
            all_congealed_imgs = torch.cat(all_congealed_imgs, dim=0)
            all_congealed_imgs_checkerboard = torch.cat(all_congealed_imgs_checkerboard, dim=0)
            average_atlas_image = all_congealed_imgs.mean(dim=0)[None, ...]
            all_vids = torch.cat((dataset.all_images, all_congealed_imgs_checkerboard.cpu(), propagated_edits_flow.cpu(),
                                  all_congealed_imgs_sim.cpu(), all_letters_on_orig.cpu()), dim=-1)
            if eval_only_mode:
                evaluation_data["all_congealed_imgs_checkerboard"] = all_congealed_imgs_checkerboard
                evaluation_data["all_unmasked_congealed_imgs"] = all_congealed_imgs
        else:
            average_atlas_image = all_congealed_imgs_sim.mean(dim=0)[None, ...]
            all_vids = torch.cat((dataset.all_images, all_congealed_imgs_sim.cpu(), all_letters_on_orig.cpu()), dim=-1)
            if eval_only_mode:
                evaluation_data["all_affine_params"] = all_affine_params
        evaluation_data[f"atlas_vis/atlas_space_average_image_unmasked"] = [wandb.Image(tensor2im(average_atlas_image))] \
            if log_to_wandb else average_atlas_image
        average_atlas_image_chk = average_atlas_image * resized_atlas_saliency + (1 - resized_atlas_saliency) * checkerboard_image
        evaluation_data[f"atlas_vis/atlas_space_average_image"] = [wandb.Image(tensor2im(average_atlas_image_chk))] \
            if log_to_wandb else average_atlas_image_chk

        evaluation_data[f"congealing/congealing_images_video"] = [
            wandb.Video(
                (255 * all_vids).to(torch.uint8), fps=1, format="mp4"
            )
        ] if log_to_wandb else all_vids

        mapping_images_grid = plot_images_grid(all_vids, nrow=all_vids.shape[0], pad_value=1, split=True, normalize=False)
        evaluation_data[f"congealing/congealing_images_grid"] = [wandb.Image(tensor2im(mapping_images_grid))] if log_to_wandb else mapping_images_grid

        return evaluation_data

    def save_locally(self, log_data, curr_batch_idx, epoch):
        if curr_batch_idx <= 0 and epoch % self.config["log_images_freq"] == 0:
            path = Path(self.config["results_folder"], str(log_data["epoch"]))
            path.mkdir(parents=True, exist_ok=True)
            for key, val in log_data.items():
                if "_im_" in key:  # per-image result
                    split_str = key.split("/")
                    save_name = split_str[-1]
                    subfolder = ''.join(split_str[:-1])
                    curr_path = path.joinpath(subfolder)
                    curr_path.mkdir(parents=True, exist_ok=True)
                else:
                    save_name = key.replace("/", "__")
                    curr_path = path
                if "atlas_keys" in key:
                    val.savefig(f"{curr_path}/{save_name}.png")
                elif "video" in key:
                    write_video(f"{curr_path}/{save_name}.mp4", val.permute(0, 2, 3, 1) * 255, fps=1, video_codec="libx264")
                elif torch.is_tensor(val):
                    save_image(val, f"{curr_path}/{save_name}.png")

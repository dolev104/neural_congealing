import torch.nn

from utils_atlas.utils_atlas_base import normalize, cosine_distance, feat_mse_loss


class AtlasLoss(torch.nn.Module):
    def __init__(self, config, dino_embedding_size, atlas_size, init_atlas_dict, device):
        super().__init__()
        self.config = config
        self.device = device
        self.dino_embedding_size = dino_embedding_size

        self.sal_loss_func = torch.nn.HuberLoss(reduction='none', delta=config["huber_delta_sal_loss"])

        if self.config["atlas_saliency_center_loss_coeff"] > 0:
            _, atlas_h, atlas_w = atlas_size
            x_grid = normalize(torch.arange(atlas_w), atlas_w, atlas_w)
            y_grid = normalize(torch.arange(atlas_h), atlas_h, atlas_h)
            self.yx_grid = torch.stack(torch.meshgrid([y_grid, x_grid], indexing='ij'), dim=0).requires_grad_(False).to(self.device)  # [2, atlas_h, atlas_w]

        # --------------------------------------------------------------------------------------------------------------
        self.imgs_idx_to_update_atlas = None
        self.all_imgs_update_atlas = True
        if self.config["ext_gradual_atlas_training"]:
            number_of_images = init_atlas_dict["all_saliency_masks"].shape[0]
            self.imgs_idx_to_update_atlas = torch.zeros(number_of_images, dtype=torch.bool)
            average_keys = init_atlas_dict["all_dino_keys"].mean(dim=0)[None, ...]
            saliency_for_keys_loss = init_atlas_dict["all_saliency_masks"].squeeze(1)

            keys_mse_loss, keys_cosine_loss = self.get_total_keys_loss(average_keys, init_atlas_dict["all_dino_keys"],
                                                                       saliency_for_keys_loss,
                                                                       self.config["keys_output_loss_coeff"],
                                                                       self.config["keys_output_cosine_loss_coeff"],
                                                                       dims_reduce_all=False)
            min_loss_im_idx = (keys_mse_loss + keys_cosine_loss).argmin()  # [number_of_images,]
            self.imgs_idx_to_update_atlas[min_loss_im_idx] = True

            self.all_imgs_update_atlas = False  # flag that says if we should note which image is updating the atlas and which is not


    @staticmethod
    def get_masked_loss(loss_func, loss_coeff, atlas_feat, imgs_feat, mask_for_loss, dims_to_reduce):
        curr_loss = (mask_for_loss * (loss_func(atlas_feat, imgs_feat)))  # batch, resy, resx
        masked_loss = loss_coeff * curr_loss.sum(dim=dims_to_reduce) / mask_for_loss.sum(dim=dims_to_reduce)  # [batch,]
        return masked_loss

    def get_per_image_masked_loss(self, loss_func, loss_coeff, atlas_feat, imgs_feat, imgs_indices, mask_for_loss):
        """
        Calaulates loss for each image separately.
        Takes into account whether an image in a batch is updating the atlas or not
        (relevant for gradual atlas training only)
        """
        all_calc_loss = []
        for i, curr_im_idx in enumerate(imgs_indices):
            if self.imgs_idx_to_update_atlas[curr_im_idx]:  # should update atlas
                curr_calc_loss = loss_func(atlas_feat, imgs_feat[i][None, ...])
            else:  # should NOT update atlas
                curr_calc_loss = loss_func(atlas_feat.detach(), imgs_feat[i][None, ...])
            all_calc_loss.append(curr_calc_loss)
        calc_loss = mask_for_loss * torch.cat(all_calc_loss, dim=0)  # [batch, resy, resx]
        masked_loss = loss_coeff * calc_loss.sum() / mask_for_loss.sum()
        return masked_loss

    def get_total_keys_loss_per_image(self, atlas_keys, imgs_keys, imgs_indices, mask_for_loss, mse_loss_coeff, cosine_loss_coeff):
        keys_mse_loss = self.get_per_image_masked_loss(feat_mse_loss, mse_loss_coeff, atlas_keys, imgs_keys, imgs_indices, mask_for_loss)
        keys_cosine_loss = self.get_per_image_masked_loss(cosine_distance, cosine_loss_coeff, atlas_keys, imgs_keys, imgs_indices, mask_for_loss)
        return keys_mse_loss, keys_cosine_loss

    def get_total_keys_loss(self, atlas_keys, imgs_keys, mask_for_loss, mse_loss_coeff, cosine_loss_coeff, dims_reduce_all=True):
        dims_to_reduce = (0, 1, 2) if dims_reduce_all else (1, 2)
        keys_mse_loss = self.get_masked_loss(feat_mse_loss, mse_loss_coeff, atlas_keys, imgs_keys, mask_for_loss, dims_to_reduce)
        keys_cosine_loss = self.get_masked_loss(cosine_distance, cosine_loss_coeff, atlas_keys, imgs_keys, mask_for_loss, dims_to_reduce)
        return keys_mse_loss, keys_cosine_loss

    @staticmethod
    def get_mag_uv_loss(delta_flow, oob_mask):
        # delta_flow [batch, resy, resx, 2]
        assert delta_flow.size(-1) == 2
        id_loss = delta_flow.pow(2)
        mask = oob_mask.permute(0, 2, 3, 1)
        return (id_loss * mask).sum() / mask.sum()

    # based on https://github.com/ykasten/layered-neural-atlases/blob/19aa32dd0cf0de7e92d279fea82844f28a15d4a0/loss_utils.py#L59
    def rigidity_loss(self, final_grid, mask_for_loss, rig_h=1):
        # rig_h: derivative amount (in pixels)
        assert final_grid.size(-1) == 2
        batch = final_grid.shape[0]
        j1_vecs = (final_grid[:, rig_h:, :, :] - final_grid[:, :-rig_h, :, :])[:, :, :-rig_h] * final_grid.shape[1] / 2  # [batch, res - rig_h, res - rig_h, 2]
        j2_vecs = (final_grid[:, :, rig_h:, :] - final_grid[:, :, :-rig_h, :])[:, :-rig_h] * final_grid.shape[1] / 2  # [batch, res - rig_h, res - rig_h, 2]
        jacobians = torch.stack((j2_vecs.reshape(batch, -1, 2), j1_vecs.reshape(batch, -1, 2)), dim=-1) / rig_h  # [batch, (res - rig_h) ** 2, 2, 2]
        JtJ = torch.einsum('bnij, bnjk -> bnik', jacobians.transpose(2, 3), jacobians)  # [batch, (res - rig_h) ** 2, 2, 2]

        a = JtJ[:, :, 0, 0] + 0.001
        b = JtJ[:, :, 0, 1]
        c = JtJ[:, :, 1, 0]
        d = JtJ[:, :, 1, 1] + 0.001

        JTJinv = torch.zeros_like(jacobians)
        JTJinv[:, :, 0, 0] = d
        JTJinv[:, :, 0, 1] = -b
        JTJinv[:, :, 1, 0] = -c
        JTJinv[:, :, 1, 1] = a
        JTJinv = JTJinv / ((a * d - b * c).unsqueeze(-1).unsqueeze(-1))  # [batch, (res - rig_h) ** 2, 2, 2]

        jacobian_loss_ = (JtJ ** 2).sum(2).sum(2).sqrt() + (JTJinv ** 2).sum(2).sum(2).sqrt()  # [batch, (res - rig_h) ** 2]

        mask = mask_for_loss[:, :-rig_h, :-rig_h].reshape(batch, -1)  # [batch, (res - rig_h) ** 2]
        jacobian_loss = (jacobian_loss_ * mask).sum() / mask.sum()
        return jacobian_loss

    def calculate_alpha_reg(self, sal_prediction):
        """
        Calculate the alpha sparsity term: linear combination between L1 and pseudo L0 penalties
        """
        l1_loss = self.config["sparsity_l1_loss_saliency_coeff"] * sal_prediction.abs().mean()  # l1_loss
        # Pseudo L0 loss using a squished sigmoid curve.
        curr_l0_loss = (torch.sigmoid(sal_prediction * 5.0) - 0.5) * 2.0
        l0_loss = self.config["sparsity_l0_loss_saliency_coeff"] * torch.mean(curr_l0_loss)  # l0_loss

        sparsity_loss = l1_loss + l0_loss
        return sparsity_loss, l1_loss, l0_loss

    def calculate_l1_loss_keys(self, predictions, mask_l1):
        total_mask = mask_l1[:, 0]
        l1_loss = (predictions.abs().sum(dim=1) * total_mask).sum() / total_mask.sum()
        return l1_loss

    def forward_stn_sim_losses(self, outputs, atlas_keys_prediction, atlas_saliency_prediction):
        # after bootstrapping, calculate only keys+saliency loss on the congealed_sim_keys+saliency
        total_loss = 0.0
        losses = {}

        congealed_sim_keys, congealed_sim_saliency = outputs["congealed_keys_sim"], outputs["congealed_saliency_sim"]

        oob_mask = outputs["congealed_white_mask_sim"].detach()
        saliency_for_keys = (atlas_saliency_prediction * oob_mask).squeeze(1)

        keys_mse_loss, keys_cosine_loss = self.get_total_keys_loss(atlas_keys_prediction, congealed_sim_keys, saliency_for_keys,
                                                                   self.config["keys_output_loss_coeff"], self.config["keys_output_cosine_loss_coeff"])
        losses["keys_mse_loss_sim"] = keys_mse_loss.item()
        losses["keys_cosine_loss_sim"] = keys_cosine_loss.item()
        total_loss += (keys_mse_loss + keys_cosine_loss)

        if self.config["atlas_saliency_loss_coeff"] > 0:
            saliency_loss = self.get_masked_loss(self.sal_loss_func, self.config["atlas_saliency_loss_coeff"],
                                                 atlas_saliency_prediction, congealed_sim_saliency, oob_mask,
                                                 dims_to_reduce=(0, 1, 2, 3))
            losses["saliency_loss_sim"] = saliency_loss.item()
            total_loss += saliency_loss
        return total_loss, losses

    def forward(self, outputs, inputs, sim_only_bootstrap=False, mapping_losses_only=False):

        total_loss = 0.
        losses = {}  # for logging
        # --------------------------------------------------------------------------------------------------------------
        keys_prediction = outputs["atlas_keys"][None, ...]  # [1, emb, resy, resx]
        saliency_prediction = outputs["atlas_saliency"][None, None, ...]  # [1, 1, resy, resx]
        if mapping_losses_only:
            keys_prediction = keys_prediction.detach()
            saliency_prediction = saliency_prediction.detach()
        assert keys_prediction.shape[1] == self.dino_embedding_size

        output_keys_gt = outputs["output_congealed_keys"]  # [batch, emb, resy, resx]
        saliency_mask_for_loss = saliency_prediction.detach()  # [1, 1, resy, resx]
        assert output_keys_gt.shape[1] == self.dino_embedding_size

        # for masking out of boundaries
        oob_mask = outputs["congealed_white_mask"].detach()  # [batch, 1, atlas_res, atlas_res]
        oob_mask_for_atlas = outputs["congealed_white_mask"].max(dim=0).values.detach()  # [1, atlas_res, atlas_res]
        # --------------------------------------------------------------------------------------------------------------

        # keys loss ----------------------------------------------------------------------------------------------------
        saliency_and_oob_mask = (saliency_mask_for_loss * oob_mask).squeeze(1)  # for masking out of boundaries
        if not mapping_losses_only and self.config["ext_gradual_atlas_training"] and not self.all_imgs_update_atlas:
            # atlas shouldn't be updated by the images that are not part of current list
            keys_mse_loss, keys_cosine_loss = self.get_total_keys_loss_per_image(keys_prediction, output_keys_gt,
                                                                                 inputs["current_im_idx"], saliency_and_oob_mask,
                                                                                 self.config["keys_output_loss_coeff"],
                                                                                 self.config["keys_output_cosine_loss_coeff"])
        else:
            keys_mse_loss, keys_cosine_loss = self.get_total_keys_loss(keys_prediction, output_keys_gt, saliency_and_oob_mask,
                                                                       self.config["keys_output_loss_coeff"],
                                                                       self.config["keys_output_cosine_loss_coeff"])
        losses["keys_mse_loss"] = keys_mse_loss.item()
        losses["keys_cosine_loss"] = keys_cosine_loss.item()
        total_loss += (keys_mse_loss + keys_cosine_loss)

        # saliency loss ------------------------------------------------------------------------------------------------
        output_saliency_gt = outputs["output_congealed_saliency"]  # [batch, 1, resy, resx]
        if self.config["atlas_saliency_loss_coeff"] > 0:
            if not mapping_losses_only and self.config["ext_gradual_atlas_training"] and not self.all_imgs_update_atlas:
                # atlas shouldn't be updated by the images that are not part of current list
                saliency_loss = self.get_per_image_masked_loss(self.sal_loss_func, self.config["atlas_saliency_loss_coeff"],
                                                               saliency_prediction, output_saliency_gt,
                                                               inputs["current_im_idx"], oob_mask)
            else:
                saliency_loss = self.get_masked_loss(self.sal_loss_func, self.config["atlas_saliency_loss_coeff"],
                                                     saliency_prediction, output_saliency_gt, oob_mask,
                                                     dims_to_reduce=(0, 1, 2, 3))
            losses["saliency_loss"] = saliency_loss.item()
            total_loss += saliency_loss

        # atlas regularization -----------------------------------------------------------------------------------------
        if not mapping_losses_only and (not self.config["ext_gradual_atlas_training"] or
                                        (self.config["ext_gradual_atlas_training"] and self.imgs_idx_to_update_atlas[inputs["current_im_idx"]].any())):
            # saliency sparsity loss
            if self.config["sparsity_loss_saliency_coeff"] > 0:
                sparsity_loss_sal, l1_loss_sal, l0_loss_sal = self.calculate_alpha_reg(saliency_prediction)
                losses["sparsity_l1_loss_atlas_saliency"] = l1_loss_sal.item()
                losses["sparsity_l0_loss_atlas_saliency"] = l0_loss_sal.item()
                sparsity_loss_saliency = self.config["sparsity_loss_saliency_coeff"] * sparsity_loss_sal
                losses["sparsity_loss_atlas_saliency"] = sparsity_loss_saliency.item()
                total_loss += sparsity_loss_saliency

            # non-salient keys sparsity loss
            if self.config["sparsity_loss_keys_coeff"] > 0:
                mask_l1 = (1 - saliency_mask_for_loss)
                sparsity_loss_keys = self.calculate_l1_loss_keys(keys_prediction, mask_l1)
                sparsity_loss_keys = self.config["sparsity_loss_keys_coeff"] * sparsity_loss_keys
                losses["sparsity_l1_loss_atlas_keys"] = sparsity_loss_keys.item()
                total_loss += sparsity_loss_keys

            # center of mass loss
            if self.config["atlas_saliency_center_loss_coeff"] > 0:
                weight_mask = oob_mask_for_atlas * saliency_prediction.squeeze(1)  # masking out of boundaries
                atlas_sal_center_of_mass = (weight_mask * self.yx_grid).sum(dim=(-1, -2)) / weight_mask.sum()

                atlas_sal_center_of_mass = (atlas_sal_center_of_mass ** 2).mean()
                atlas_sal_center_loss = self.config["atlas_saliency_center_loss_coeff"] * atlas_sal_center_of_mass
                losses["atlas_saliency_center_loss"] = atlas_sal_center_loss.item()
                total_loss += atlas_sal_center_loss

        # STN regularization -------------------------------------------------------------------------------------------
        _, affine_params, delta_flow, final_grid = outputs["transformation_params"]  # final_grid is the final composed sampling grid

        # STNsim loss --------------------------------------------------------------------------------------------------
        losses["transformation_params"] = affine_params.detach()  # [batch, 4]
        # loss on affine params
        if self.config["affine_scale_loss_coeff"] > 0:
            scale = affine_params[:, 1]  # [batch]
            scale_loss = ((1 - scale).abs() ** 2).mean()
            curr_scale_loss = self.config["affine_scale_loss_coeff"] * scale_loss
            losses["affine_scale_loss"] = curr_scale_loss.item()
            total_loss += curr_scale_loss

        if not sim_only_bootstrap:
            # STNflow loss ---------------------------------------------------------------------------------------------
            if self.config["mag_uv_loss_coeff"] > 0:
                curr_mag_uv_loss = self.config["mag_uv_loss_coeff"] * self.get_mag_uv_loss(delta_flow, oob_mask)
                losses["mag_uv_loss"] = curr_mag_uv_loss.item()
                total_loss += curr_mag_uv_loss

            if self.config["local_rigidity_loss_coeff"] > 0:
                curr_local_rigidity_loss = self.config["local_rigidity_loss_coeff"] * self.rigidity_loss(final_grid, saliency_and_oob_mask, rig_h=1)
                losses["local_rigidity_loss"] = curr_local_rigidity_loss.item()
                total_loss += curr_local_rigidity_loss

            if self.config["global_rigidity_loss_coeff"] > 0:
                curr_global_rigidity_loss = self.config["global_rigidity_loss_coeff"] * self.rigidity_loss(final_grid, oob_mask.squeeze(1), rig_h=self.config["global_rig_derivative"])
                losses["global_rigidity_loss"] = curr_global_rigidity_loss.item()
                total_loss += curr_global_rigidity_loss

            # separate objective after bootstrapping  ------------------------------------------------------------------
            # continue applying keys + saliency losses on the STNsim outputs
            total_loss_stn_sim, losses_stn_sim = self.forward_stn_sim_losses(outputs["congealed_sim_keys_sal"], keys_prediction.detach(), saliency_prediction.detach())
            total_loss += total_loss_stn_sim
            losses.update(losses_stn_sim)

        # --------------------------------------------------------------------------------------------------------------

        losses["total_loss"] = total_loss.item()
        return total_loss, losses

import torch

from models.spatial_transformers.spatial_transformer import STNWrapper


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AtlasModel(torch.nn.Module):
    def __init__(self, config, dino_keys_embedding_size, init_atlas_dict, device, init_with_flow=False):
        super().__init__()
        self.config = config
        self.device = device
        self.atlas_resolution = self.config["atlas_resolution"]

        self.stn = STNWrapper(config, self.atlas_resolution, device, init_with_flow=init_with_flow)
        self.with_flow = self.stn.with_flow

        self.atlas_size = (dino_keys_embedding_size + 1, self.atlas_resolution, self.atlas_resolution)  # + 1 for saliency
        # initialize atlas
        init_atlas_keys = init_atlas_dict["all_dino_keys"].mean(dim=0)
        all_saliencies = init_atlas_dict["all_saliency_masks"].mean(dim=0)
        clamped_saliencies = (((all_saliencies - all_saliencies.min()) * (0.9 - 0.1)) / (all_saliencies.max() - all_saliencies.min())) + 0.1  # turn to range [0.1, 0.9]
        imgs_saliency_masks = torch.log(clamped_saliencies / (1 - clamped_saliencies))  # need logit (before sigmoid is applied)
        init_atlas_saliency = imgs_saliency_masks.squeeze()

        if "atlas_params_requires_grad" in config and not config["atlas_params_requires_grad"]:
            self.atlas_keys, self.atlas_saliency = None, None  # experiment not supported in main code at the moment
        else:
            init_atlas_keys.requires_grad_(True)
            self.atlas_keys = torch.nn.Parameter(init_atlas_keys)
            init_atlas_saliency.requires_grad_(True)
            self.atlas_saliency = torch.nn.Parameter(init_atlas_saliency)

    def apply_bw_warp(self, input_image, bw_sim_mat, flow, padding_mode, img_size=None):
        return self.stn.apply_bw_warp(input_image, bw_sim_mat, flow, padding_mode, img_size=img_size)

    def congeal_points(self, img, points, normalize_input_points=True, unnormalize_output_points=False,
                       output_resolution=None, return_full=False):
        return self.stn.congeal_points(img, points, normalize_input_points=normalize_input_points,
                                       unnormalize_output_points=unnormalize_output_points,
                                       output_resolution=output_resolution, return_full=return_full)

    def uncongeal_points(self, img, points_congealed, samp_grid=None, unnormalize_output_points=True,
                         normalize_input_points=False, output_resolution=None, return_congealed_img=False):
        return self.stn.uncongeal_points(img, points_congealed, samp_grid=samp_grid,
                                         unnormalize_output_points=unnormalize_output_points,
                                         normalize_input_points=normalize_input_points,
                                         output_resolution=output_resolution, return_congealed_img=return_congealed_img)

    def get_atlas_params(self):
        atlas_keys = self.atlas_keys  # [emb, h, w]
        atlas_saliency = torch.sigmoid(self.atlas_saliency)  # [h, w]
        return atlas_keys, atlas_saliency

    def process_batch(self, input_image, input_keys=None, input_saliency=None, im_output_res=None,
                      padding_mode_im='border', padding_mode_keys='border', padding_mode_saliency='zeros',
                      return_congealed_keys=False, return_congealed_sal=False, return_oob_mask=False,
                      return_stn_sim_outputs=False, return_images=False):
        outputs = self.stn(input_image, input_keys, input_saliency, im_output_res, padding_mode_im=padding_mode_im,
                           padding_mode_keys=padding_mode_keys, padding_mode_saliency=padding_mode_saliency,
                           return_congealed_keys=return_congealed_keys, return_congealed_sal=return_congealed_sal,
                           return_oob_mask=return_oob_mask, return_stn_sim_outputs=return_stn_sim_outputs,
                           return_images=return_images)
        return outputs

    def add_stn_flow_to_training(self):
        self.stn.add_stn_flow_to_training()
        self.with_flow = self.stn.with_flow

    def get_mapping_model(self):
        return self.stn.get_mapping_model()

    def forward(self, input_dict):
        outputs = {}

        ret = self.process_batch(input_dict["input_image"], input_keys=input_dict["input_keys"],
                 input_saliency=input_dict["input_saliency"], padding_mode_im=self.config["stn_padding_mode_im"],
                 padding_mode_keys=self.config["stn_padding_mode_keys"], padding_mode_saliency=self.config["stn_padding_mode_saliency"],
                 return_congealed_keys=True, return_congealed_sal=True, return_oob_mask=True,
                 return_stn_sim_outputs=self.with_flow, return_images=False)
        if not self.with_flow:
            transformation_params, congealed_keys, congealed_saliency, congealed_white_mask = ret
        else:
            transformation_params, congealed_keys, congealed_saliency, congealed_white_mask, congealed_sim_keys_sal = ret
            outputs["congealed_sim_keys_sal"] = congealed_sim_keys_sal

        outputs["atlas_keys"], outputs["atlas_saliency"] = self.get_atlas_params()

        outputs["output_congealed_keys"] = congealed_keys  # [batch, emb, h, w]
        outputs["output_congealed_saliency"] = congealed_saliency  # [batch, 1, h, w]
        outputs["congealed_white_mask"] = congealed_white_mask  # for masking out of boundaries
        outputs["transformation_params"] = transformation_params  # transformation_params = (bw_sim_mat, affine_params, delta_flow, final_grid)

        return outputs


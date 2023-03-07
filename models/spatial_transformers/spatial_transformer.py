# Code taken from GANgealing (https://github.com/wpeebles/gangealing/tree/main/models/spatial_transformers)
# same architecture, modified code a bit and added an STNWrapper
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.stylegan2.networks import EqualLinear, ConvLayer, ResBlock
from models.spatial_transformers.warping_heads import FlowHead, SimilarityHead
from models.spatial_transformers.antialiased_sampling import BilinearDownsample
from models import requires_grad


# Code taken from GANgealing (https://github.com/wpeebles/gangealing/tree/main/models/spatial_transformers)
def unravel_index(indices, shape):
    # https://stackoverflow.com/a/65168284
    r"""Converts flat indices into unraveled coordinates in a target shape.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).
    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord, dim=-1)

    return coord


class STNWrapper(nn.Module):
    def __init__(self, config, atlas_resolution, device, init_with_flow=False):
        super().__init__()
        self.config = config
        self.atlas_resolution = atlas_resolution  # should also be the resolution of image keys and saliency
        transforms = ["similarity", "flow"]
        self.stn = ComposedSTN(transforms, flow_size=config["flow_size"], supersize=config["real_size"],
                               channel_multiplier=config["stn_channel_multiplier"],
                               stn_head_antialias=config["stn_head_antialias"])

        # for masking out of boundaries
        self.white_mask = torch.ones((1, 1, self.atlas_resolution, self.atlas_resolution), requires_grad=False, device=device)

        if self.config["bootstrap_stn_sim"] > 0 and not init_with_flow:
            self.with_flow = False
            self.stn.stns[self.stn.transforms.index('flow')].eval()
            requires_grad(self.stn.stns[self.stn.transforms.index('flow')], False)
        else:
            self.with_flow = True

    def forward(self, input_image, input_keys, input_saliency, im_output_res, padding_mode_im='border',
                padding_mode_keys='border', padding_mode_saliency='zeros', return_congealed_keys=False,
                return_congealed_sal=False, return_oob_mask=False, return_stn_sim_outputs=False, return_images=False):
        final_congealed_image, transformations = self.stn(input_image, output_resolution=im_output_res,
                                                          sim_only=(not self.with_flow), return_flow=True,
                                                          padding_mode=padding_mode_im,
                                                          return_final_congealed_image=return_images)
        if self.with_flow:
            _, bw_sim_mat, affine_params, final_sampling_grid, delta_flow, _ = transformations
        else:
            _, bw_sim_mat, affine_params = transformations
            final_sampling_grid, delta_flow = None, None

        ret = [(bw_sim_mat, affine_params, delta_flow, final_sampling_grid)]
        if return_congealed_keys:
            congealed_keys = self.apply_bw_warp(input_keys, bw_sim_mat, final_sampling_grid, padding_mode_keys,
                                                img_size=(*input_keys.shape[:2], self.atlas_resolution, self.atlas_resolution))
            ret.append(congealed_keys)
        if return_congealed_sal:
            congealed_saliency = self.apply_bw_warp(input_saliency, bw_sim_mat, final_sampling_grid, padding_mode_saliency,
                                                    img_size=(*input_saliency.shape[:2], self.atlas_resolution, self.atlas_resolution))
            ret.append(congealed_saliency)
        if return_oob_mask:
            # for masking out of boundaries
            curr_white_mask = self.white_mask.expand(bw_sim_mat.shape[0], -1, -1, -1)
            congealed_white_mask = self.apply_bw_warp(curr_white_mask, bw_sim_mat, final_sampling_grid, 'zeros',
                                                      img_size=(*curr_white_mask.shape[:2], self.atlas_resolution, self.atlas_resolution))
            ret.append(congealed_white_mask)
        if return_stn_sim_outputs:  # used only during training
            congealed_sim_keys_sal = {}
            congealed_sim_keys_sal["congealed_keys_sim"] = self.apply_bw_warp(input_keys, bw_sim_mat, None,
                padding_mode_keys, img_size=(*input_keys.shape[:2], self.atlas_resolution, self.atlas_resolution))
            congealed_sim_keys_sal["congealed_saliency_sim"] = self.apply_bw_warp(input_saliency, bw_sim_mat, None,
                padding_mode_saliency, img_size=(*input_saliency.shape[:2], self.atlas_resolution, self.atlas_resolution))
            curr_white_mask = self.white_mask.expand(bw_sim_mat.shape[0], -1, -1, -1)  # for masking out of boundaries
            congealed_sim_keys_sal["congealed_white_mask_sim"] = self.apply_bw_warp(curr_white_mask, bw_sim_mat, None,
                'zeros', img_size=(*curr_white_mask.shape[:2], self.atlas_resolution, self.atlas_resolution))
            ret.append(congealed_sim_keys_sal)
        if return_images:  # used only during logging/eval
            ret.append(final_congealed_image)
            if self.with_flow:
                congealed_sim = self.apply_bw_warp(input_image, bw_sim_mat, None, padding_mode_keys, img_size=input_image.size())
            else:
                congealed_sim = None
            ret.append(congealed_sim)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def apply_bw_warp(self, input_features, bw_sim_mat, flow, padding_mode, img_size=None):
        return self.stn.apply_bw_warp(input_features, bw_sim_mat, flow, padding_mode, img_size=img_size)

    def congeal_points(self, img, points, normalize_input_points=True, unnormalize_output_points=False,
                       output_resolution=None, return_full=False):
        return self.stn.congeal_points(img, points, normalize_input_points=normalize_input_points,
                                       unnormalize_output_points=unnormalize_output_points,
                                       output_resolution=output_resolution, return_full=return_full)

    def uncongeal_points(self, img, points_congealed, samp_grid=None, unnormalize_output_points=True, normalize_input_points=False,
                         output_resolution=None, return_congealed_img=False):
        return self.stn.uncongeal_points(img, points_congealed, gridB=samp_grid,
                                         unnormalize_output_points=unnormalize_output_points, normalize_input_points=normalize_input_points,
                                         output_resolution=output_resolution, return_congealed_img=return_congealed_img)

    def add_stn_flow_to_training(self):
        self.with_flow = True
        requires_grad(self.stn.stns[self.stn.transforms.index('flow')], True)
        self.stn.stns[self.stn.transforms.index('flow')].train()

    def get_mapping_model(self):
        return self.stn


# Code taken from GANgealing (https://github.com/wpeebles/gangealing/tree/main/models/spatial_transformers)
# same architecture, modified code a bit
class ComposedSTN(nn.Module):

    """
    Chains a sequence of STNs together by composing warps. This module provides some connective tissue
    to let STNs that perform different warps (e.g., similarity and per-pixel flow) talk with each other.
    This class has only been tested with transforms=['similarity', 'flow']. Chaining multiple unique similarity STNs
    (optionally followed by a final flow STN) should likely work as well. A flow STN followed by a similarity STN is not
    currently supported and will not work without some light modifications to SimilarityHead. Similarly, chaining
    multiple unique flow STNs together will not work without some modifications to FlowHead.
    """

    def __init__(self, transforms, flow_size=128, supersize=256, channel_multiplier=0.5,
                 stn_head_antialias=True):
        super().__init__()
        stns = []
        for transform in transforms:
            stns.append(SpatialTransformer(flow_size, supersize, transform=transform,
                                           channel_multiplier=channel_multiplier, flow_downsample=8,
                                           stn_head_antialias=stn_head_antialias))
        if transforms != ['similarity', 'flow']:
            print('WARNING: ComposedSTN is only tested for transforms=["similarity", "flow"].')
        self.stns = nn.ModuleList(stns)
        self.transforms = transforms[:]  # Deepcopy
        self.stn_in_size = flow_size
        self.N_minus_1 = len(self.stns) - 1
        self.is_flow = 'flow' in transforms
        if self.is_flow:
            self.identity_flow = self.stns[transforms.index('flow')].identity_flow

    def apply_bw_warp(self, input_img, bw_sim_mat, flow, padding_mode, img_size=None):
        if self.is_flow and flow is not None:
            congealed_im = self.stns[self.transforms.index('flow')].apply_bw_warp(input_img, bw_sim_mat, flow, padding_mode, img_size=img_size)
        else:
            congealed_im = self.stns[self.transforms.index('similarity')].apply_bw_warp(input_img, bw_sim_mat, flow, padding_mode, img_size=img_size)
        return congealed_im

    def forward(self, input_img, return_warp=None, return_flow=False, output_resolution=None,
                input_img_for_sampling=None, padding_mode='border', sim_only=False,
                return_final_congealed_image=True, **stn_forward_kwargs):
        """
        :param return_warp: If True, returns the final (N, H, W, 2) sampling grid produced by the composed STN
        :param return_flow: If True, returns either the final (N, K, 2, 3) affine matrix (if the final network is a
                            similarity STN) or the final (N, H, W, 2) residual flow (if the final network is a flow STN)
        :param input_img_for_sampling: If specified, the STN will sample from this image instead of input_img above.
        :param sim_only: If True, will only do the similarity STN's forward pass.
        :param return_final_congealed_image: If True, the unconstrained flow STN will warp the image using the
                                             final (composed) sampling grid.
        :param stn_forward_kwargs: Any additional arguments that should be used in each STN's forward pass
        :return: (N, C, H, W) output warped images, as well as any additional requested outputs
        See spatial_transformers.warping_heads.SimilarityHead for documentation of the other inputs.
        """
        out = input_img
        source_pixels = input_img if input_img_for_sampling is None else input_img_for_sampling
        warp = None  # None corresponds to identity warp
        intermediate_output_resolution = self.stn_in_size
        transformations = []
        for i, stn in enumerate(self.stns):
            output_resolution_t = output_resolution if (i == self.N_minus_1 or sim_only) else intermediate_output_resolution

            stn_out = stn(out, return_warp=False, return_flow=True,
                          input_img_for_sampling=source_pixels, base_warp=warp,
                          output_resolution=output_resolution_t, pack=True,
                          padding_mode=padding_mode, output_final_congealed_image=return_final_congealed_image, **stn_forward_kwargs)
            out, all_transformations = stn_out
            grid, flow_or_matrix, grid_or_affine_params = all_transformations
            transformations.extend([grid, flow_or_matrix, grid_or_affine_params])

            # TODO: Currently, flow --> similarity and flow --> flow is not supported
            if not sim_only and i == 0:  # detach STNsim from STNflow
                warp = flow_or_matrix.detach()
                out = out.detach()
            else:
                warp = flow_or_matrix
            if sim_only:
                break

        ret = [out]
        if return_warp:
            ret.append(grid)
        if return_flow:
            ret.append(transformations)
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def uncongeal_points(self, imgB, points_congealed, gridB=None, output_resolution=None, unnormalize_output_points=True,
                         normalize_input_points=False, return_congealed_img=False, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points from the congealed space to imgA.
        """
        assert imgB.size(0) == points_congealed.size(0)
        if normalize_input_points:
            points_congealed = SpatialTransformer.normalize(points_congealed, imgB.size(-1), self.stn_in_size)
        if gridB is None:
            congealed_img, gridB = self.forward(imgB, return_warp=True, return_flow=False, output_resolution=output_resolution,
                                                **stn_forward_kwargs)
        pointsB = F.grid_sample(gridB.permute(0, 3, 1, 2), points_congealed.unsqueeze(2).float(), padding_mode='border').squeeze(3).permute(0, 2, 1)
        if unnormalize_output_points:
            pointsB = SpatialTransformer.unnormalize(pointsB, imgB.size(-1), imgB.size(-1))  # Back to coordinates in [0, H-1]
        if return_congealed_img:
            return pointsB, congealed_img
        else:
            return pointsB

    def congeal_points(self, imgA, pointsA, output_resolution=None, normalize_input_points=True,
                          unnormalize_output_points=False, return_full=False, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points from the congealed space to imgA.
        """
        assert imgA.size(0) == pointsA.size(0)
        intermediate_output_resolution = self.stn_in_size
        outA = imgA
        points_congealed = pointsA
        warpA = None
        for i, stn in enumerate(self.stns):  # Compose in forward order
            output_resolution_t = output_resolution if i == self.N_minus_1 else intermediate_output_resolution
            norm_input_points = normalize_input_points if i == 0 else True
            unnorm_out_points = unnormalize_output_points if i == self.N_minus_1 else True
            outA, warpA, points_congealed = stn.congeal_points(outA, points_congealed, normalize_input_points=norm_input_points,
                                                           unnormalize_output_points=unnorm_out_points,
                                                           output_resolution=output_resolution_t, base_warp=warpA,
                                                           input_img_for_sampling=imgA, return_full=True,
                                                           **stn_forward_kwargs)
        if return_full:
            return outA, warpA, points_congealed
        else:
            return points_congealed

    def transfer_points(self, imgA, imgB, pointsA, output_resolution=None, congeal_kwargs={}, uncongeal_kwargs={},
                        **stn_forward_kwargs):
        """
        Given input images imgA, transfer known key points pointsA to target images imgB.
        """
        assert imgA.size(0) == imgB.size(0) == pointsA.size(0)
        # Step 1: Map the key points in imgA to the congealed image (forward warp):
        points_congealed = self.congeal_points(imgA, pointsA, output_resolution=output_resolution,
                                               normalize_input_points=True, **congeal_kwargs,
                                               **stn_forward_kwargs)
        # Step 2: Map the key points in the congealed image to those in imgB (reverse warp):
        pointsB = self.uncongeal_points(imgB, points_congealed, output_resolution=output_resolution,
                                        normalize_input_points=True, unnormalize_output_points=True,
                                        **uncongeal_kwargs, **stn_forward_kwargs)
        return pointsB

    def load_single_state_dict(self, state_dict, index, strict=True):
        # state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith(f'stns.{index}')}
        # print(state_dict.keys())
        return self.stns[index].load_state_dict(state_dict, strict)

    def load_several_state_dicts(self, state_dicts, indices, strict=True):
        assert len(state_dicts) == len(indices)
        for state_dict, index in zip(state_dicts, indices):
            self.load_single_state_dict(state_dict, index, strict)

    def load_state_dict(self, state_dict, strict=True):
        ignore = ['warp_head.one_hot']
        for i in range(len(self.stns)):
            ignore.extend([f'stns.{i}.input_downsample.kernel_horz', f'stns.{i}.input_downsample.kernel_vert',
                           f'stns.{i}.warp_head.rebias'])
        ignore = set(ignore)
        filtered = {k: v for k, v in state_dict.items() if k not in ignore}
        return super().load_state_dict(filtered, False)


# Code taken from GANgealing (https://github.com/wpeebles/gangealing/tree/main/models/spatial_transformers)
# same architecture, modified code a bit
class SpatialTransformer(nn.Module):
    def __init__(self, flow_size, supersize, channel_multiplier=0.5, blur_kernel=[1, 3, 3, 1],
                 transform='similarity', flow_downsample=8, stn_head_antialias=True):
        """
        Here is how the SpatialTransformer works:
            (1) Take an image as input.
            (2) Regress a flow field from this input image at some fixed resolution (flow_size, flow_size),
                usually flow_size=128.
            (3) Optionally upsample/downsample this flow field with bilinear interpolation. Note that bilinear
                 upsampling the flow field will usually result in a very high quality output image (much higher quality
                 than bilinearly resizing the warped output image after-the-fact).
            (4) Sample the input image with the (optionally resized) flow field to obtain the warped output image.
        :param flow_size: The resolution of the flow field produced by the Spatial Transformer and also the resolution
                      at which images are processed by the warp parameter regression portion of the STN.
        :param supersize: The resolution of input images to the Spatial Transformer. Should be >= flow_size.
        :param channel_multiplier: Controls the number of channels in the conv layers
        :param blur_kernel: Low-pass filter to lightly anti-alias intermediate activations of the network.
        :param transform: Class of transformations produced by the STN (either 'similarity' or 'flow')
        :param flow_downsample: Only relevant when transform = 'flow'. As part of Step (2) above, a low resolution
                flow is initially regressed at resolution (flow_size / flow_downsample, flow_size / flow_downsample),
                before being upsampled to (flow_size, flow_size) via learned convex upsampling as described in the RAFT
                paper. Should be a power of 2.
        :param stn_head_antialias: If True, will apply spatial transforms with mipmap anti-aliasing, else with grid_sample.
        """
        super().__init__()
        if supersize > flow_size:
            self.input_downsample = BilinearDownsample(supersize // flow_size, 3)

        self.input_downsample_required = supersize > flow_size
        self.stn_in_size = flow_size
        self.is_flow = transform == 'flow'

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, int(channels[flow_size]), 1)]

        log_size = int(math.log(flow_size, 2))
        log_downsample = int(math.log(flow_downsample, 2))

        in_channel = channels[flow_size]

        end_log = log_size - 4 if self.is_flow else 2
        assert end_log >= 0

        num_downsamples = 0
        for i in range(log_size, end_log, -1):
            downsample = (not self.is_flow) or (num_downsamples < log_downsample)
            num_downsamples += downsample
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(int(in_channel), int(out_channel), blur_kernel, downsample))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        if not self.is_flow:
            self.final_linear = EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu')

        if transform == 'similarity':
            warp_class = SimilarityHead
            in_shape = channels[4]
        elif transform == 'flow':
            warp_class = FlowHead
            in_shape = (1, in_channel, flow_size // flow_downsample, flow_size // flow_downsample)
        else:
            raise NotImplementedError
        self.warp_head = warp_class(in_shape, antialias=stn_head_antialias, flow_downsample=flow_downsample)
        if self.is_flow:
            self.identity_flow = self.warp_head.identity_flow

    def apply_bw_warp(self, input_img, flow_or_matrix, flow, padding_mode, img_size=None):
        if self.is_flow and flow is not None:
            out = self.warp_head.apply_bw_warp(input_img, flow, self.warp_head.warper, padding_mode, img_size=img_size)
            return out
        else:
            out = self.warp_head.apply_bw_warp(input_img, flow_or_matrix, self.warp_head.warper, padding_mode, img_size=img_size)
            return out[-1]

    def get_bw_grid_sim(self, input_img, flow_or_matrix, padding_mode, img_size=None):
        out = self.warp_head.get_bw_grid_sim(input_img, flow_or_matrix, self.warp_head.warper, padding_mode, img_size=img_size)
        return out[0]

    def forward(self, input_img, output_resolution=None, return_warp=False, return_flow=False,
                padding_mode='border', input_img_for_sampling=None, base_warp=None,
                pack=False, output_final_congealed_image=True):
        """
        :param input_img: (N, C, H, W) input image tensor used to regress a warp. If input_img_for_sampling (see below)
                           is NOT specified, then input_img will also be used as the source for sampling pixels
                           according the STN's predicted warp.
        :param output_resolution: int (or None). This will be the size of the output warped image and the predicted flow
                                  field. Internally, this bilinearly resizes the flow field, and thus yields a much
                                  higher quality warped output image compared to bilinear resizing in pixel space.
        :param return_warp: If True, returns the final (composed) (N, H, W, 2) sampling grid regressed by the STN
        :param return_flow: This argument's behavior varies depending on the type of WarpHead. If this is a similarity
                            STN, this will return a tuple of: the sampling grid, an (N, 2, 3) tensor of affine matrices
                            representing the similarity transformation and a (N, 4) tensor of the transformation
                            parameters. Otherwise, if this is an unconstrained flow STN, this will return a tuple of:
                            the final (composed) sampling grid, a (N, H, W, 2) tensor representing the _residual_ flow
                            field predicted by the STN, and the final sampling grid in the resolution determined by
                            output_final_congealed_image.
        :param input_img_for_sampling: (N, C, H*, W*) input image tensor. If specified, the STN will sample from this
                                        image instead of input_img above. This argument is useful if, e.g., you have a
                                        high resolution version of input_img; then you can pass the high resolution
                                        image here to get a higher quality output warped image.
        :param pack: If True, will return all outputs given by the WarpHeads.
        :param output_final_congealed_image: If True, will warp the input image.
        For explanations of all other inputs, please refer to the documentation of warping_heads.SimilarityHead.forward.

        :return: (N, C, H, W) tensor of congealed output images. Additional outputs will be returned if return_warp,
                 return_flow, or pack is True.
        """

        if input_img.size(-1) > self.stn_in_size:
            regression_input = self.input_downsample(input_img)
        else:
            regression_input = input_img

        if input_img_for_sampling is not None:
            source_pixels = input_img_for_sampling
        else:
            source_pixels = input_img

        out = self.convs(regression_input)

        batch, channel, height, width = out.shape
        out = self.final_conv(out)

        if not self.is_flow:
            out = out.view(batch, -1)
            out = self.final_linear(out)  # [batch, 512]

        output_resolution = output_resolution if output_resolution is not None else self.stn_in_size
        out, grid, M, grid_or_bw_mat_params = self.warp_head(source_pixels, out, output_resolution=output_resolution, base_warp=base_warp,
                                           padding_mode=padding_mode, output_final_congealed_image=output_final_congealed_image)
        if pack:  # Return everything
            return out, (grid, M, grid_or_bw_mat_params)
        else:  # TODO: could make argument-packing more elegant:
            rtn = [out]
            if return_warp:
                rtn.append(grid)
            if return_flow:
                rtn.append((grid, M, grid_or_bw_mat_params))
            if len(rtn) == 1:
                rtn = rtn[0]
            return rtn

    @staticmethod
    def normalize01(points, res, out_res):
        return points.div(out_res - 1).mul((res - 1) / res)

    @staticmethod
    def unnormalize01(points, res, out_res):
        return points.div((res - 1) / res).mul(out_res - 1)

    @staticmethod
    def normalize(points, res, out_res):
        return points.div(out_res - 1).add(-0.5).mul(2).mul((res - 1) / res)

    @staticmethod
    def unnormalize(points, res, out_res):
        return points.div((res - 1) / res).div(2).add(0.5).mul(out_res - 1)

    @staticmethod
    def convert(points, current_res, target_res):
        points = SpatialTransformer.normalize(points, target_res, current_res)
        points = SpatialTransformer.unnormalize(points, target_res, target_res)
        return points

    def congeal_points(self, imgA, pointsA, normalize_input_points=True, unnormalize_output_points=False, output_resolution=None,
                       input_img_for_sampling=None, return_full=False, **stn_forward_kwargs):
        assert imgA.size(0) == pointsA.size(0)
        N = imgA.size(0)
        num_points = pointsA.size(1)
        source_res = imgA.size(-1) if input_img_for_sampling is None else input_img_for_sampling.size(-1)
        outA, (_, flow_or_matrixA, _) = self.forward(imgA, return_warp=False, return_flow=True,
                                                    output_resolution=output_resolution,
                                                    input_img_for_sampling=input_img_for_sampling,
                                                    **stn_forward_kwargs)
        if normalize_input_points:
            pointsA = self.normalize(pointsA, source_res, source_res)  # [0, H-1] --> [-1, 1]
        # Forward similarity transform (nice and closed-form):
        if not self.is_flow:
            points_congealed = self.apply_bw_mat_on_uncongealed_pnts(pointsA, flow_or_matrixA, num_points, N)
            if unnormalize_output_points:
                points_congealed = self.unnormalize(points_congealed, source_res, source_res)
        # While we have access to the reverse (inverse) flows since our STN using reverse sampling, we do NOT have
        # access to the forward flows. To approximate the A --> congealed forward flow,
        # we roughly reverse the congealed --> A inverse flow (which we have access to) via a brute-force nearest
        # neighbor search on the sampling grid.
        else:
            assert flow_or_matrixA.size(-1) == 2  # flow_or_matrixA is a flow field with shape (N, H, W, 2)
            gridA = flow_or_matrixA + self.identity_flow
            gridA_reshaped = gridA.reshape(N, gridA.size(1), gridA.size(2), 1, 1, 2)  # (N, H, W, 1, 1, 2)
            pointsA = pointsA.reshape(N, 1, 1, num_points, 2, 1)  # (N, 1, 1, num_points, 2, 1)
            similarities = (gridA_reshaped @ pointsA)[..., 0, 0]  # (N, H, W, num_points)
            # Compute distances as: ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
            distances = pointsA.pow(2).squeeze(-1).sum(dim=-1) + gridA_reshaped.pow(2).sum(dim=-1).squeeze(-1) - 2 * similarities
            # TODO: currently, key points that get forward mapped beyond the congealed image boundary are clamped to
            # TODO: the border. There's probably a better way to handle this problem...
            nearest_neighbors = distances.reshape(N, gridA_reshaped.size(1) * gridA_reshaped.size(2), num_points).argmin(
                dim=1)  # (N, num_points)
            points_congealed = unravel_index(nearest_neighbors, (gridA_reshaped.size(1), gridA_reshaped.size(2)))  # (N, num_points, 2)
        if return_full:
            return outA, flow_or_matrixA, points_congealed
        else:
            return points_congealed

    @staticmethod
    def apply_bw_mat_on_uncongealed_pnts(pointsA, flow_or_matrixA, num_points, N):
        pointsA = torch.cat([pointsA, torch.ones(N, num_points, 1, device=pointsA.device)], 2)  # (N, |points|, 3)
        onehot = torch.tensor([[[0, 0, 1]]], dtype=torch.float, device=flow_or_matrixA.device).repeat(N, 1, 1)
        matrixA_3x3 = torch.cat([flow_or_matrixA, onehot], 1)
        A2congealed = torch.inverse(matrixA_3x3).permute(0, 2, 1)
        points_congealed = (pointsA @ A2congealed)[
            ..., [0, 1]]  # Apply transformation and remove homogeneous coordinate
        return points_congealed

    def uncongeal_points(self, imgB, points_congealed, unnormalize_output_points=True, normalize_input_points=False,
                         output_resolution=None, input_img_for_sampling=None, flow_or_matrixB=None, gridB=None, **stn_forward_kwargs):
        """
        Given input images imgB, transfer known key points points_congealed to target images imgB.
        """
        assert imgB.size(0) == points_congealed.size(0)
        N = imgB.size(0)
        num_points = points_congealed.size(1)
        source_res = imgB.size(-1) if input_img_for_sampling is None else input_img_for_sampling.size(-1)
        if flow_or_matrixB is None and gridB is None:
            outB, gridB, flow_or_matrixB = self.forward(imgB, return_warp=True, return_flow=True,
                                                        output_resolution=output_resolution,
                                                        input_img_for_sampling=input_img_for_sampling,
                                                        **stn_forward_kwargs)

        # Compose the forward similarity transform (A --> congealed) and inverse similarity transform (congealed --> B)
        # This is the easier path since we there is a nice closed-form solution:
        if normalize_input_points:
            points_congealed = self.normalize(points_congealed, source_res, imgB.size(-1))
        if not self.is_flow:
            pointsB = self.uncongeal_sim_pnts(points_congealed, flow_or_matrixB, N, num_points)
        # Compose the forward flow (A --> congealed) and the inverse flow (congealed --> B)
        # While we have access to the inverse flows since our STN using reverse sampling, we do NOT have
        # access to the forward flows which complicates things. To approximate the A --> congealed forward flow,
        # we roughly reverse the congealed --> A inverse flow (which we have access to) via a brute-force nearest
        # neighbor search on the sampling grid.
        else:
            assert gridB.size(-1) == 2  # gridB is a flow field with shape (N, H, W, 2)
            pointsB = F.grid_sample(gridB.permute(0, 3, 1, 2), points_congealed.unsqueeze(2).float(), padding_mode='border').squeeze(3).permute(0, 2, 1)
        if unnormalize_output_points:
            pointsB = self.unnormalize(pointsB, imgB.size(-1), source_res)  # Back to coordinates in [0, H-1]
        return pointsB

    @staticmethod
    def uncongeal_sim_pnts(points_congealed, flow_or_matrixB, N, num_points):
        onehot = torch.tensor([[[0, 0, 1]]], dtype=torch.float, device=flow_or_matrixB.device).repeat(N, 1, 1)
        points_congealed = torch.cat([points_congealed, torch.ones(N, num_points, 1, device=points_congealed.device)],
                                     2)  # (N, |points|, 3)
        congealed2B = torch.cat([flow_or_matrixB, onehot], 1).permute(0, 2, 1)
        pointsB = (points_congealed @ congealed2B)[..., [0, 1]]  # Apply transformation and remove homogeneous coordinate
        return pointsB

    def transfer_points(self, imgA, imgB, pointsA, output_resolution=None, **stn_forward_kwargs):
        """
        Given input images imgA, transfer known points pointsA to target images imgB.
        """
        assert imgA.size(0) == imgB.size(0) == pointsA.size(0)
        # Step 1: Map the key points in imgA to the congealed image (forward warp):
        points_congealed = self.congeal_points(imgA, pointsA, output_resolution=output_resolution,
                                           **stn_forward_kwargs)
        # Step 2: Map the key points in the congealed image to those in imgB (reverse warp):
        pointsB = self.uncongeal_points(imgB, points_congealed, output_resolution=output_resolution,
                                    normalize_input_points=False, **stn_forward_kwargs)
        return pointsB

    def load_state_dict(self, state_dict, strict=True):
        ignore = {'warp_head.one_hot', 'warp_head.rebias', 'input_downsample.kernel_horz',
                  'input_downsample.kernel_vert'}
        filtered = {k: v for k, v in state_dict.items() if k not in ignore}
        return super().load_state_dict(filtered, False)

# Code taken from GANgealing (https://github.com/wpeebles/gangealing/tree/main/models/spatial_transformers)
# same architecture, modified code a bit
"""
Contains output heads useful for building Spatial Transformer Networks (STNs). These output heads
take an input image (and its features) as input, and then regress and apply a warp to the input image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.spatial_transformers.antialiased_sampling import MipmapWarp, Warp
from models.stylegan2.networks import EqualConv2d


class SimilarityHead(nn.Module):

    # Regresses and applies a similarity warp (rotation, uniform scale, vertical shift and horizontal shift)

    def __init__(self, in_shape, antialias=True, **kwargs):
        """
        :param in_shape: int. SimiarlityHead.forward expects input features of shape (N, D). Pass D here.
        :param antialias: boolaen. Whether or not to use antialiasing when applying the similarity transformation.
        :param kwargs: Ignore; this is used to filter any parameters that are only processed by FlowHead
        """
        super().__init__()

        self.num_warp_params = 4  # rotation, uniform scale, horizontal shift, vertical shift
        self.linear = nn.Linear(in_shape, self.num_warp_params, bias=True)
        # Initialize so linear always produces the identity transform in the first forward pass:
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()
        # Select a pixel sampling algorithm:
        self.warper = MipmapWarp(max_num_levels=3.5) if antialias else Warp()
        self.register_buffer('one_hot', torch.tensor([0, 0, 1], dtype=torch.float).view(1, 1, 1, 3))

    @staticmethod
    def make_affine_matrix(rot, scale, shift_x, shift_y):
        # This function takes the raw output of the parameter regression network and converts them into
        # an affine matrix representing the predicted similarity transformation.
        # Inputs each of size (N, 1)
        N, _ = rot.size()
        rot = torch.tanh(rot) * math.pi
        scale = torch.exp(scale)
        cos_rot = torch.cos(rot)
        sin_rot = torch.sin(rot)
        matrix = [scale * cos_rot, -scale * sin_rot, shift_x,
                  scale * sin_rot, scale * cos_rot, shift_y]
        matrix = torch.cat(matrix, dim=1)  # (N, 6)
        matrix = matrix.reshape(N, 2, 3)  # (N, 2, 3)
        return matrix, torch.cat((rot, scale, shift_x, shift_y), dim=1)

    def forward(self, img, features, output_resolution=None, padding_mode='border', **kwargs):
        """
        :param img: (N, C, H*, W*) input image that the STN will sample from when producing the warped output image.
        :param features: (N, D) features from which a similarity transformation will be predicted.
        :param output_resolution: int. The intermediate flow field will be bilinearly resampled to this resolution. If
                                  None, no resizing is performed on the flow.
        :param padding_mode: ['border'|'reflection'|'zero']. Controls how the STN extrapolates pixels when sampling
                             beyond image boundaries.
        :return: A tuple of outputs:
                    (N, C, H, W) tensor, the warped output images.
                    (N, H, W, 2) tensor, the predicted similarity reverse sampling grid
                    (N, 2, 3) tensor, the predicted affine matrices.
                    (N, 4) tensor, the predicted similarity transformation parameters (theta, scale, tx, ty)
        """

        params = self.linear(features)  # [batch, 4] Regress raw similarity warp parameters from input features

        params = torch.split(params, 1, dim=1)
        matrix, affine_params = self.make_affine_matrix(*params)
        if output_resolution is None:
            img_size = torch.Size([img.size(0), *img.size()[1:]])
        else:
            img_size = torch.Size([img.size(0), img.size(1), output_resolution, output_resolution])

        grid, out = self.apply_bw_warp(img, matrix, self.warper, padding_mode, img_size=img_size)
        return out, grid, matrix, affine_params

    @staticmethod
    def apply_bw_warp(img, matrix, warper, padding_mode='border', img_size=None):
        img_size = img_size if img_size is not None else img.size()
        grid = F.affine_grid(matrix, img_size, align_corners=False)
        out = warper(img, grid, padding_mode=padding_mode)
        return grid, out


class FlowHead(nn.Module):

    # Regresses and applies an arbitrary transformation via reverse sampling

    def __init__(self, in_shape, antialias=True, flow_downsample=8, **kwargs):
        super().__init__()

        self.flow_downsample = flow_downsample
        self.identity_flow = self.initialize_flow(in_shape).cuda()
        # This output head will produce a raw flow field at (128 / flow_downsample, 128 / flow_downsample) resolution:
        self.flow_out = nn.Sequential(EqualConv2d(in_shape[1], in_shape[1], 3, padding=1),
                                      nn.ReLU(),
                                      EqualConv2d(in_shape[1], 2, 3, padding=1))
        # This ensures the output will initially be the identity transformation:
        nn.init.zeros_(self.flow_out[-1].weight)
        nn.init.zeros_(self.flow_out[-1].bias)
        # Mask head used for convex upsampling of the raw flow field (as done in RAFT)
        self.mask_out = nn.Sequential(EqualConv2d(in_shape[1], in_shape[1], 3, padding=1),
                                      nn.ReLU(),
                                      EqualConv2d(in_shape[1], 9 * flow_downsample * flow_downsample, 3, padding=1))
        self.warper = MipmapWarp(max_num_levels=3.5) if antialias else Warp()

    def initialize_flow(self, in_shape):
        N, C, H, W = in_shape
        # Identity sampling grid:
        coords = F.affine_grid(torch.eye(2, 3).unsqueeze(0), (N, C, self.flow_downsample * H, self.flow_downsample * W))
        return coords

    def upsample_flow(self, flow, mask):  # RAFT: https://github.com/princeton-vl/RAFT/blob/master/core/update.py
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, H, W, _ = flow.shape
        flow = flow.permute(0, 3, 1, 2)  # NHW2 --> N2HW
        mask = mask.view(N, 1, 9, self.flow_downsample, self.flow_downsample, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.flow_downsample * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 4, 2, 5, 3, 1)
        up_flow = up_flow.reshape(N, self.flow_downsample*H, self.flow_downsample*W, 2)
        return up_flow

    def compute_flow(self, features):
        flow = self.flow_out(features)
        N, _, H, W = flow.size()
        flow = flow.reshape(N, 1, 2, H, W)
        flow = flow.permute(0, 1, 3, 4, 2)

        mask = self.mask_out(features)
        mask = mask.reshape(N, 1, 9 * self.flow_downsample * self.flow_downsample, H, W)
        return flow, mask

    def forward(self, img, features, output_resolution=None, base_warp=None, padding_mode='border',
                output_final_congealed_image=True):
        """
        :param img: (N, C, H*, W*) input image that the STN will sample from when producing the warped output image.
        :param features: (N, D, H*, W*) features from which a flow field will be predicted.
        :param output_resolution: int. The intermediate flow field will be bilinearly resampled to this resolution. If
                                  None, no resizing is performed on the flow.
        :param base_warp: (N, 2, 3) tensor representing an affine warp. If specified, composes the warp predicted
                          by this function with base_warp.
        :param padding_mode: ['border'|'reflection'|'zero']. Controls how the STN extrapolates pixels when sampling
                             beyond image boundaries.
        :param output_final_congealed_image: If True, will warp the input image.
        :return: A tuple of outputs:
                    (N, C, H, W) tensor, the warped output images if output_final_congealed_image, else None
                    (N, H*, W*, 2) tensor, the fully-composed flow field in the resolution of the atlas
                    (N, 128, 128, 2) tensor, the residual flow field predicted by this STN
                    (N, H*, W*, 2) tensor, the fully-composed flow field used for sampling the input image
        """
        low_res_delta_flow, mask = self.compute_flow(features)
        N, _, H, W, _ = low_res_delta_flow.size()

        low_res_delta_flow = low_res_delta_flow.reshape(N, H, W, 2)
        mask = mask.reshape(N, -1, H, W)
        delta_flow = self.upsample_flow(low_res_delta_flow, mask)
        flow = self.identity_flow + delta_flow
        if base_warp is not None:  # TODO: This currently assumes that base_warp is a similarity transform
            flow = apply_affine(base_warp, flow)

        if output_final_congealed_image:  # relevant only during logging/evaluation
            flow_high_res = flow
            if output_resolution is None:
                img_size = torch.Size([img.size(0), img.size(1), flow.size(1), flow.size(2)])
            else:
                img_size = torch.Size([img.size(0), img.size(1), output_resolution, output_resolution])
                # interpolating the flow will yield a much higher quality output than interpolating pixels:
                if output_resolution != flow.size(1):
                    flow_high_res = F.interpolate(flow.permute(0, 3, 1, 2), scale_factor=output_resolution / flow.size(2), mode='bilinear').permute(0, 2, 3, 1)
            out = self.apply_bw_warp(img, flow_high_res, self.warper, padding_mode, img_size=img_size)
        else:
            out = None
            flow_high_res = flow
        return out, flow, delta_flow, flow_high_res

    @staticmethod
    def apply_bw_warp(img, grid, warper, padding_mode='border', img_size=None):
        if img_size is not None and img_size[-1] != grid.size(2):
            grid = F.interpolate(grid.permute(0, 3, 1, 2), scale_factor=img_size[-1] / grid.size(2),
                                 mode='bilinear').permute(0, 2, 3, 1)
        out = warper(img, grid, padding_mode=padding_mode)
        return out


def apply_affine(matrix, grid):
    # This function is similar to torch.nn.functional.affine_grid, except it applies the
    # input affine matrix to an arbitrary input sampling grid instead of the identity sampling grid
    grid_size = grid.size()
    grid = grid.reshape(grid.size(0), -1, 2)
    ones = torch.ones(grid.size(0), grid.size(1), 1, device=grid.device)
    grid = torch.cat([grid, ones], 2)
    warped = grid @ matrix.permute(0, 2, 1)
    warped = warped.reshape(grid_size)
    return warped

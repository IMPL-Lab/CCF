# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS


@NECKS.register_module()
class SECONDFPNV2(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """

    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 num_attached_conv=0,
                 conv_kwargs=dict(kernel_size=3, dilation=2, padding=2, stride=1),
                 conv_in_channel=64,
                 conv_out_channel=64,
                 init_cfg=None):
        # if for GroupNorm,
        # cfg is dict(type='GN', num_groups=num_groups, eps=1e-3, affine=True)
        super(SECONDFPNV2, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(norm_cfg, out_channel)[1],
                                    nn.ReLU(inplace=True))
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if num_attached_conv > 0:
            conv_list = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                if i > 0:
                    conv_in_channel = conv_out_channel
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel,
                    out_channels=conv_out_channel,
                    **conv_kwargs_i,
                )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list.append(convnormrelu)

            conv_list1 = []
            for i in range(num_attached_conv):

                if isinstance(conv_kwargs, dict):
                    conv_kwargs_i = conv_kwargs
                elif isinstance(conv_kwargs, list):
                    assert len(conv_kwargs) == num_attached_conv
                    conv_kwargs_i = conv_kwargs[i]

                # if i > 0:
                #     conv_in_channel = conv_out_channel//2
                conv = build_conv_layer(
                    conv_cfg,
                    in_channels=conv_in_channel//2,
                    out_channels=conv_out_channel//2,
                    **conv_kwargs_i,
                )

                if norm_cfg is None:
                    convnormrelu = nn.Sequential(
                        conv,
                        nn.ReLU(inplace=True)
                    )
                else:
                    convnormrelu = nn.Sequential(
                        conv,
                        build_norm_layer(norm_cfg, conv_out_channel//2)[1],
                        nn.ReLU(inplace=True)
                    )
                conv_list1.append(convnormrelu)

            self.dense_conv = nn.ModuleList(conv_list1)
            self.dense_conv_1 = nn.ModuleList(conv_list)
            self.dense_conv_2 = nn.ModuleList(conv_list)
        else:
            self.dense_conv = None
            self.dense_conv_1 = None

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # batch_idx =  torch.zeros_like(batch_x)
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    @auto_fp16()
    def forward(self, x, swin_format=False, insfusion_layer=None, **kwargs):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """

        assert len(x) == len(self.in_channels)

        if not swin_format:
            ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

            # if 'sst_backbone' in kwargs:
            #     batch_size = ups[0].shape[0]
            #     bev_coords = self.create_dense_coord(ups[0].shape[-1], ups[0].shape[-2], batch_size).type_as(ups[0]).int()
            #     this_coords = []
            #     for k in range(batch_size):
            #         this_coord = bev_coords[k]
            #         this_coord = this_coord.reshape(4, -1).transpose(1, 0)
            #         this_coords.append(this_coord)
            #     return_coords = torch.cat(this_coords, dim=0)
            #     x0, x1 = ups
            #     x0 = x0.flatten(2, 3).permute(0, 2, 1).reshape(-1, x0.shape[1])
            #     x1 = x1.flatten(2, 3).permute(0, 2, 1).reshape(-1, x0.shape[1])
            #     x0 = kwargs['sst_encoder'][0](x0, return_coords, batch_size)
            #     x0 = kwargs['sst_backbone'][0](x0)[0]
            #     x1 = kwargs['sst_encoder'][1](x1, return_coords, batch_size)
            #     x1 = kwargs['sst_backbone'][1](x1)[0]
            #     ups = [x0, x1]

            if insfusion_layer is not None:
                ups, kwargs = insfusion_layer(bev_query=ups, key=None, value=None, **kwargs)

            if kwargs.get('test_visual_mode', False):
                return ups, kwargs

            if len(ups) > 1:
                out = torch.cat(ups, dim=1)
            else:
                out = ups[0]

            if self.dense_conv is not None:
                for conv in self.dense_conv:
                    temp = conv(out)
                    if temp.shape == out.shape:
                        out = temp + out
                    else:
                        out = temp

        else:

            if self.dense_conv is not None:
                for conv in self.dense_conv:
                    temp = conv(x[0])
                    if temp.shape == x[0].shape:
                        x[0] = temp + x[0]
                    else:
                        x[0] = temp

            if self.dense_conv_1 is not None:
                for conv in self.dense_conv_1:
                    temp = conv(x[1])
                    if temp.shape == x[1].shape:
                        x[1] = temp + x[1]
                    else:
                        x[1] = temp

            if self.dense_conv_2 is not None:
                for conv in self.dense_conv_2:
                    temp = conv(x[2])
                    if temp.shape == x[2].shape:
                        x[2] = temp + x[2]
                    else:
                        x[2] = temp

            ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

            if len(ups) > 1:
                out = torch.cat(ups, dim=1)
            else:
                out = ups[0]

        out = out.permute(0, 1, 3, 2).contiguous() #todo debug bevfusion

        if insfusion_layer is not None:
            return [out], kwargs

        return [out]


        # #post sst
        # batch_size = out.shape[0]
        # return_features = out.reshape(batch_size, out.shape[1], -1).permute(0, 2, 1).reshape(-1, out.shape[1])
        #
        # bev_coords = self.create_dense_coord(180, 180, batch_size).type_as(return_features).int()
        # # this_features = []
        # this_coords = []
        # for k in range(batch_size):
        #     this_coord = bev_coords[k]
        #     this_coord = this_coord.reshape(4,-1).transpose(1,0)
        #     # this_feature = spatial_features[k].reshape([C * D, -1])[:, unique.long()]
        #     # return_features.append(this_features)
        #     this_coords.append(this_coord)
        #
        # # return_features = torch.cat(this_features, dim=0)
        # return_coords = torch.cat(this_coords, dim=0)

        # return return_features, return_coords






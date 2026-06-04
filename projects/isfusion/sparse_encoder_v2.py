# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import auto_fp16
from torch import nn as nn
import torch
import random

from mmdet3d.ops import SparseBasicBlock, make_sparse_convmodule
# spconv v1
# from mmdet3d.ops import spconv as spconv

from mmdet3d.models.builder import MIDDLE_ENCODERS

from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:  # spconv v1
    from spconv.pytorch import SparseConvTensor, SparseSequential
else:
    from mmcv.ops import SparseConvTensor, SparseSequential


def sparse_normalization_perturbation(sparse_tensor, alpha_std=0.75, beta_std=0.75):
    """Apply normalization perturbation to sparse tensor features.
    
    Args:
        sparse_tensor: SparseConvTensor with features (N, C)
        alpha_std: Standard deviation for alpha perturbation
        beta_std: Standard deviation for beta perturbation
    
    Returns:
        SparseConvTensor: Perturbed sparse tensor
    """
    features = sparse_tensor.features  # (N, C)
    
    # Calculate mean across spatial dimensions (per channel)
    feat_mean = features.mean(dim=0, keepdim=True)  # (1, C)
    
    # Generate perturbation parameters
    ones_mat = torch.ones_like(feat_mean)
    alpha = torch.normal(ones_mat, alpha_std * ones_mat)  # (1, C)
    beta = torch.normal(ones_mat, beta_std * ones_mat)   # (1, C)
    
    # Apply perturbation: alpha * feat - alpha * feat_mean + beta * feat_mean
    perturbed_features = alpha * features - alpha * feat_mean + beta * feat_mean
    
    # Create new sparse tensor with perturbed features
    if IS_SPCONV2_AVAILABLE:
        from spconv.pytorch import SparseConvTensor
    else:
        from mmcv.ops import SparseConvTensor
    
    return SparseConvTensor(
        perturbed_features,
        sparse_tensor.indices,
        sparse_tensor.spatial_shape,
        sparse_tensor.batch_size
    )


@MIDDLE_ENCODERS.register_module()
class SparseEncoderV2(nn.Module):
    r"""Sparse encoder for SECOND and Part-A2.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        order (list[str]): Order of conv module. Defaults to ('conv',
            'norm', 'act').
        norm_cfg (dict): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        base_channels (int): Out channels for conv_input layer.
            Defaults to 16.
        output_channels (int): Out channels for conv_out layer.
            Defaults to 128.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
            Defaults to ((16, ), (32, 32, 32), (64, 64, 64), (64, 64, 64)).
        block_type (str): Type of the block to use. Defaults to 'conv_module'.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 block_type='conv_module',
                 use_norm_perturb=False,
                 norm_perturb_prob=0.5,
                 norm_perturb_stages=None,
                 **kwargs):
        super(SparseEncoderV2, self).__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        
        # Normalization perturbation parameters
        self.use_norm_perturb = use_norm_perturb
        self.norm_perturb_prob = norm_perturb_prob
        # Default to apply perturbation at all encoder stages
        self.norm_perturb_stages = norm_perturb_stages or list(range(len(encoder_channels)))

        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':  # pre activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = make_sparse_convmodule(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule(  # autoalign
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')


    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size, swin_format=False, img_feats=None, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4), \
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """

        coors = coors.int()  # must be type int
        input_sp_tensor = SparseConvTensor(voxel_features, coors,          # spconv 2
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        encode_features.append(x)
        for i, encoder_layer in enumerate(self.encoder_layers):
            x = encoder_layer(x)
            
            # Apply normalization perturbation if enabled
            if (self.use_norm_perturb and self.training and 
                i in self.norm_perturb_stages and 
                random.random() < self.norm_perturb_prob):
                x = sparse_normalization_perturbation(x)
            
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features, encode_features, kwargs


    # spconv 2.0
    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_cfg (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.
            block_type (str, optional): Type of the block to use.
                Defaults to 'conv_module'.
            conv_cfg (dict, optional): Config of conv layer. Defaults to
                dict(type='SubMConv3d').

        Returns:
            int: The number of encoder output channels.
        """
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlock(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
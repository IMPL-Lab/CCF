# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import torch_scatter

from ..builder import QUERY_GENERATORS


@QUERY_GENERATORS.register_module()
class PointCloudQueryGeneratorBEV(BaseModule):
    '''Generate point query features from BEV features.'''
    def __init__(self, in_channels=128, hidden_channel=128, pts_use_cat=False,
                 dataset='nuscenes', virtual_voxel_size=None, point_cloud_range=None, head_pc_range=None):
        super(PointCloudQueryGeneratorBEV, self).__init__()

        assert dataset == 'nuscenes'

        # a shared convolution
        self.empty_pos = nn.Embedding(100000, 3)
        self.empty_embed = nn.Embedding(1, hidden_channel)

        self.pre_bev_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.bev_embed = nn.Identity()
        self.query_embed = nn.Sequential(
            nn.Linear(in_channels, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
            nn.LayerNorm(hidden_channel, hidden_channel),
        )
        self.query_pred_embed = nn.Sequential(
            nn.Linear(7 * 32, hidden_channel),
            nn.ReLU(),
            nn.Linear(hidden_channel, hidden_channel)
        )

        self.pts_use_cat = pts_use_cat
        if self.pts_use_cat:
            num_cls = 10
            self.pts_cat_embed = nn.Embedding(num_cls, hidden_channel)

        self.virtual_voxel_size = virtual_voxel_size
        self.point_cloud_range = point_cloud_range
        self.head_pc_range = head_pc_range

    def init_weights(self):
        super(PointCloudQueryGeneratorBEV, self).init_weights()
        nn.init.uniform_(self.empty_pos.weight.data, 0, 1)
        self.empty_pos.weight.requires_grad = False

    @staticmethod
    def pos2embed(pos, num_pos_feats=128, temperature=10000):
        import math
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
        pos_x = pos[..., None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        return pos_x.flatten(-2)

    def forward(self, lidar_feat, query_feat, query_xyz, query_pred, query_cat, batch_size):
        # assert not any(x.requires_grad for x in query_xyz) and not any(x.requires_grad for x in query_pred) and \
        #        not any(x.requires_grad for x in query_cat) and not any(x.requires_grad for x in query_feat)

        device = lidar_feat.device
        voxel_size = torch.tensor(self.virtual_voxel_size, device=device)
        pc_range = torch.tensor(self.point_cloud_range, device=device)

        B, H, W, C = lidar_feat.shape
        lidar_feat = lidar_feat.view(B, H*W, C)  # [B, H*W, C]
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device)
        )
        bev_xyz = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [H*W, 2]
        bev_xyz = bev_xyz.unsqueeze(0).repeat(B, 1, 1)  # [B, H*W, 2]
        
        bev_lidar_feat = self.pre_bev_embed(lidar_feat) + lidar_feat
        lidar_feat = bev_lidar_feat
        lidar_xyz = bev_xyz

        # generate query content features from 3D detection
        # query_feat comes from raw proposals, while query_xyz/query_pred may be filtered
        # by bbox decoding. Align lengths per batch to avoid shape mismatches.
        aligned_query_feat = []
        aligned_query_pred = []
        aligned_query_xyz = []
        aligned_query_cat = []
        for b in range(batch_size):
            cur_feat = query_feat[b]
            cur_pred = query_pred[b]
            cur_xyz = query_xyz[b]
            cur_cat = query_cat[b]
            valid_len = min(len(cur_feat), len(cur_pred), len(cur_xyz), len(cur_cat))
            aligned_query_feat.append(cur_feat[:valid_len])
            aligned_query_pred.append(cur_pred[:valid_len])
            aligned_query_xyz.append(cur_xyz[:valid_len])
            aligned_query_cat.append(cur_cat[:valid_len])

        query_feat = [x if len(x) > 0 else x.new_zeros((0, 128)) for x in aligned_query_feat]
        query_pred = aligned_query_pred
        query_xyz = aligned_query_xyz
        query_cat = aligned_query_cat

        query_feat = [self.query_embed(x) + x for x in query_feat]
        query_feat_w_pred = [
            x + self.query_pred_embed(self.pos2embed(pred, 32, temperature=20))
            for x, pred in zip(query_feat, query_pred)
        ]
        if self.pts_use_cat:
            query_feat_w_pred = [x + self.pts_cat_embed(cat) for x, cat in zip(query_feat_w_pred, query_cat)]
        query_feat = query_feat_w_pred

        feat_size = lidar_feat.size(-1)
        lidar_size = [H*W] * B
        max_size = H*W

        # pad key/value positions
        bev_grid_range = torch.tensor([0, 0, 0, W, H, 1], device=device)
        head_pc_range = torch.tensor(self.head_pc_range, device=device)
        # normalize grid coordinates to [0,1]
        lidar_xyz_norm = lidar_xyz / torch.tensor([W, H], device=device)
        lidar_pos = lidar_feat.new_zeros([batch_size, max_size, 2]) + 0.5
        pad_size = min(max_size, self.empty_pos.weight.size(0))
        lidar_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :2]
        for b in range(batch_size):
            lidar_pos[b, :lidar_size[b]] = lidar_xyz_norm[b]

        # pad key/value features (actually no padding in BEV, because size is fixed)
        # lidar_feat_in = lidar_feat

        # pad query positions
        max_query_size = max(len(x) for x in query_feat)
        query_pos = lidar_feat.new_zeros([batch_size, max_query_size, 3])
        pad_size = min(max_query_size, self.empty_pos.weight.size(0))
        
        query_pos[:, -pad_size:] = self.empty_pos.weight[:pad_size, :3] * (
                head_pc_range[3:6] - head_pc_range[0:3]) + head_pc_range[0:3]
        for b in range(batch_size):
            query_pos[b, :len(query_feat[b])] = query_xyz[b][..., :3]

        # pad query content feats
        query_feat_in = lidar_feat.new_zeros([batch_size, max_query_size, feat_size]) + self.empty_embed.weight
        for b in range(batch_size):
            if len(query_feat[b]) > 0:
                query_feat_in[b, :len(query_feat[b])] = query_feat[b]
        query_feat = query_feat_in

        return lidar_feat, lidar_pos, query_feat, query_pos
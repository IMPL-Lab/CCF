# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from StreamPETR (https://github.com/exiawsh/StreamPETR)
# Copyright (c) Shihao Wang
# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import copy
import math
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from mmcv.cnn import Linear, bias_init_with_prob, ConvModule
from mmcv.cnn.bricks.transformer import build_transformer_layer

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox
from mmcv.ops.box_iou_rotated import box_iou_rotated
from mmdet3d.core import nms_bev
from mmdet3d.core.bbox.structures import xywhr2xyxyr

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, \
    nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, \
    SELayer_Linear

from projects.mmdet3d_plugin.models.utils.sigma_reparam import remove_all_normalization_layers, convert_to_sn

def pos2embed(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = 2 * (dim_t // 2) / num_pos_feats + 1
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


@HEADS.register_module()
class MV2DFusionHeadV2Decouple(AnchorFreeHead):
    _version = 2
    TRACKING_CLASSES = ['car', 'truck', 'bus', 'trailer', 'motorcycle', 'bicycle', 'pedestrian']

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                #  memory_len=6 * 256,
                 topk_proposals=256,
                #  num_propagated=256,
                 with_dn=True,
                #  with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 transformer=None,
                 normedlinear=False,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)), ),
                 test_cfg=dict(max_per_img=100),
                 # denoise config
                 scalar=5,
                 noise_scale=0.4,
                 noise_trans=0.0,
                 dn_weight=1.0,
                 split=0.5,
                 # image query config
                 prob_bin=50,
                 # nms config
                 post_bev_nms_thr=0.2,
                 post_bev_nms_score=0.0,
                 post_bev_nms_ops=[],
                 # init config
                 init_cfg=None,
                 # IoU calculator config
                 iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                 # Saved field control
                 save_fields=['pred_history', 'modal', 'query_id'],
                 sigma_reparam=False,
                 modal_weight=False,
                 dynamic_q_zero_init=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is MV2DFusionHeadV2Decouple):
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']

            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            iou_sampler_cfg = dict(type='IoUThresholdPseudoSampler', iou_threshold=0.1)
            self.iou_sampler = build_sampler(iou_sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.topk_proposals = topk_proposals
        self.with_dn = with_dn
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.embed_dims = embed_dims

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split

        self.act_cfg = transformer.get('act_cfg', dict(type='ReLU', inplace=True))
        self.num_pred = transformer['decoder']['num_layers']
        self.normedlinear = normedlinear
        self.prob_bin = prob_bin
        
        # Sigma Reparameterization
        self.sigma_reparam = sigma_reparam

        # Modal Weight
        self.modal_weight = modal_weight
        self.dynamic_q_zero_init = dynamic_q_zero_init

        super(MV2DFusionHeadV2Decouple, self).__init__(num_classes, in_channels, init_cfg=init_cfg)

        # IoU calculator
        self.iou_calculator = build_iou_calculator(iou_calculator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        if self.sigma_reparam:
            # Replace all LayerNorm modules with spectral normalization.
            # self.transformer = remove_all_normalization_layers(convert_to_sn(self.transformer, linear_init_gain=1.0))
            self.transformer = convert_to_sn(self.transformer, linear_init_gain=1.0)
        
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        # nms config
        self.post_bev_nms_thr = post_bev_nms_thr
        self.post_bev_nms_score = post_bev_nms_score
        self.post_bev_nms_ops = post_bev_nms_ops

        self._init_layers()

        self.fp16_enabled = False

        self.save_fields = save_fields

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])
        self.reference_points = nn.Embedding(self.num_query, 3)
        # if self.num_propagated > 0:
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(14, use_ln=False)

        # image distribution query positional encoding
        prob_bin = self.prob_bin
        self.dyn_q_embed = nn.Embedding(1, self.embed_dims)
        self.dyn_q_enc = MLN(256)
        self.dyn_q_pos = nn.Sequential(
            nn.Linear(prob_bin * 3, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )
        self.dyn_q_pos_with_prob = SELayer_Linear(self.embed_dims, in_channels=prob_bin)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, prob_bin))
        reg_branch = nn.Sequential(*reg_branch)
        self.dyn_q_prob_branch = nn.ModuleList([
            copy.deepcopy(reg_branch) for _ in range(self.num_pred)
        ])

        # point cloud embedding
        self.pts_embed = nn.Sequential(
            nn.Linear(128, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.pts_query_embed = nn.Sequential(
            nn.Linear(128, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.pts_q_embed = nn.Embedding(1, self.embed_dims)


    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @staticmethod
    def transform3d(pose, coords3d):
        coords3d = torch.cat([coords3d, torch.ones_like(coords3d[..., 0:1])], dim=-1)   # B, ..., 4
        shape = coords3d.shape[:-1]
        new_shape = [shape[i] if i == 0 else 1 for i in range(len(shape))]
        pose = pose.view(*new_shape, 4, 4)
        transformed_coords3d = (pose @ coords3d[..., None])[..., :3, 0]
        return transformed_coords3d

    @staticmethod
    def rotate2d(pose, coords2d):
        shape = coords2d.shape[:-1]
        new_shape = [shape[i] if i == 0 else 1 for i in range(len(shape))]
        pose = pose.view(*new_shape, 4, 4)[..., :2, :2]
        rotated_coords2d = (pose @ coords2d[..., None])[..., 0]
        return rotated_coords2d

    @staticmethod
    def get_box_info(bbox_preds):
        bbox_x, bbox_y, bbox_w, bbox_l, bbox_o = bbox_preds[..., 0], bbox_preds[..., 1], bbox_preds[..., 3], \
                                                 bbox_preds[..., 4], bbox_preds[..., 6]
        bbox_z, bbox_h = bbox_preds[..., 2], bbox_preds[..., 5]
        # bbox_o = -(bbox_o + np.pi / 2)
        bbox_o = (bbox_o + np.pi / 2)
        center = torch.stack([bbox_x, bbox_y], dim=-1)
        cos, sin = torch.cos(bbox_o), torch.sin(bbox_o)
        pc0 = torch.stack([bbox_x + cos * bbox_l / 2 + sin * bbox_w / 2,
                           bbox_y + sin * bbox_l / 2 - cos * bbox_w / 2], dim=-1)
        pc1 = torch.stack([bbox_x + cos * bbox_l / 2 - sin * bbox_w / 2,
                           bbox_y + sin * bbox_l / 2 + cos * bbox_w / 2], dim=-1)
        pc2 = 2 * center - pc0
        pc3 = 2 * center - pc1

        xyxyo = torch.stack([pc0, pc1, pc2, pc3, center], dim=-2)   # [..., 5, 2]
        bbox_z = bbox_z[..., None, None].expand_as(xyxyo[..., :1])
        xyxyo = torch.cat([xyxyo, bbox_z], dim=-1)
        return xyxyo, torch.stack([bbox_w, bbox_l, bbox_h], dim=-1), torch.stack([cos, sin], dim=-1)

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [
                torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),
                          dim=1) for img_meta in img_metas]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            # gt_num
            known_num = [t.size(0) for t in targets]

            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0),), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            # groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                               diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (
                            self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes

            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            if reference_points.dim() == 2:
                padded_reference_points = \
                    torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3:
                padded_reference_points = torch.cat([padding_bbox.unsqueeze(0).repeat(batch_size, 1, 1), reference_points], dim=1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(
                    reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
        else:
            if reference_points.dim() == 2:
                padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            elif reference_points.dim() == 3:
                padded_reference_points = reference_points
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is MV2DFusionHeadV2Decouple:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)

    def gen_dynamic_query(self, static_query, dynamic_query, dynamic_query_feats=None):
        B = len(dynamic_query)
        zero = static_query.sum() * 0
        max_len = max(x.size(0) for x in dynamic_query)
        max_len = max(max_len, 1)
        query_coords = static_query.new_zeros((B, max_len, dynamic_query[0].size(1), 3))
        query_probs = static_query.new_zeros((B, max_len, dynamic_query[0].size(1)))
        query_ref = static_query.new_zeros((B, max_len, 3)) + zero + 0.5
        query_mask = static_query.new_zeros((B, max_len), dtype=torch.bool)
        query_feats = static_query.new_zeros((B, max_len, self.embed_dims))
        self.num_query = max_len

        for b in range(B):
            dyn_q = dynamic_query[b][..., :3].clone()
            dyn_q[..., 0:3] = (dyn_q[..., 0:3] - self.pc_range[0:3]) / (
                    self.pc_range[3:6] - self.pc_range[0:3])
            dyn_q_prob = dynamic_query[b][..., 3]
            ref_point = (dyn_q_prob[:, None] @ dyn_q)[:, 0]
            query_coords[b, :dyn_q.size(0)] = dyn_q
            query_probs[b, :dyn_q.size(0)] = dyn_q_prob
            query_ref[b, :dyn_q.size(0)] = ref_point
            query_mask[b, :dyn_q.size(0)] = 1
            if dynamic_query_feats is not None:
                query_feats[b, :dyn_q.size(0)] = dynamic_query_feats[b][:dyn_q.size(0)]

        return query_ref, query_coords, query_probs, query_feats, query_mask

    def gen_pts_query(self, pts_query_center):
        pts_ref = pts_query_center.clone()
        pts_ref[..., 0:3] = (pts_ref[..., 0:3] - self.pc_range[0:3]) / (
                self.pc_range[3:6] - self.pc_range[0:3])
        self.num_query += pts_ref.size(1)
        return pts_ref

    def forward(self, img_metas, dyn_query=None, dyn_feats=None,
                pts_query_center=None, pts_query_feat=None, pts_feat=None, pts_pos=None, mode='all', eval_decoupled=False, 
                **data):

        # process image feats
        intrinsics = data['intrinsics'] / 1e3
        extrinsics = data['extrinsics'][..., :3, :]
        mln_input = torch.cat([intrinsics[..., 0,0:1], intrinsics[..., 1,1:2], extrinsics.flatten(-2)], dim=-1)
        mln_input = mln_input.flatten(0, 1).unsqueeze(1)
        mlvl_feats = data['img_feats_for_det']
        B, N, _, _, _ = mlvl_feats[0].shape
        feat_flatten_img = []
        spatial_flatten_img = []
        for i in range(1, len(mlvl_feats)):
            B, N, C, H, W = mlvl_feats[i].shape
            mlvl_feat = mlvl_feats[i].reshape(B * N, C, -1).transpose(1, 2)
            mlvl_feat = self.spatial_alignment(mlvl_feat, mln_input)
            feat_flatten_img.append(mlvl_feat.to(torch.float))
            spatial_flatten_img.append((H, W))
        feat_flatten_img = torch.cat(feat_flatten_img, dim=1)
        spatial_flatten_img = torch.as_tensor(spatial_flatten_img, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index_img = torch.cat((spatial_flatten_img.new_zeros((1, )), spatial_flatten_img.prod(1).cumsum(0)[:-1]))

        # process point cloud feats
        feat_flatten_pts = self.pts_embed(pts_feat)
        pos_flatten_pts = pts_pos

        # generate queries based on mode (training) or always both (inference)
        num_query_img, num_query_pts = 0, 0
        query_coords, query_probs, query_feats, query_mask = None, None, None, None
        
        # For training, respect the mode parameter for decoupled loss
        # For inference, behavior depends on eval_decoupled parameter
        if self.training or eval_decoupled:
            if mode in ['img', 'all'] and dyn_query is not None:
                reference_points_img, query_coords, query_probs, query_feats, query_mask_img = \
                    self.gen_dynamic_query(self.reference_points.weight, dyn_query, dyn_feats.get('query_feats', None))
                num_query_img = self.num_query
            
            # Reset num_query for pts query generation if needed
            self.num_query = 0

            if mode in ['pts', 'all'] and pts_query_center is not None:
                pts_ref = self.gen_pts_query(pts_query_center)
                num_query_pts = self.num_query
            
            self.num_query = num_query_img + num_query_pts

            if mode == 'all':
                query_mask = torch.cat([torch.ones_like(pts_ref[..., 0]).bool(), query_mask_img], dim=1)
                reference_points = torch.cat([pts_ref, reference_points_img], dim=1)
            elif mode == 'pts':
                reference_points = pts_ref
                query_mask = torch.ones_like(pts_ref[..., 0]).bool()
                # Create empty tensors for img queries to avoid errors
                query_coords = reference_points.new_zeros(B, 0, self.prob_bin, 3)
                query_probs = reference_points.new_zeros(B, 0, self.prob_bin)
                query_feats = reference_points.new_zeros(B, 0, self.embed_dims)
            elif mode == 'img':
                reference_points = reference_points_img
                query_mask = query_mask_img
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            # Inference with eval_decoupled=False: always generate both queries (current behavior)
            reference_points_img, query_coords, query_probs, query_feats, query_mask_img = \
                self.gen_dynamic_query(self.reference_points.weight, dyn_query, dyn_feats.get('query_feats', None))

            # generate point cloud query
            pts_ref = self.gen_pts_query(pts_query_center)
            num_query_pts = pts_ref.size(1)
            
            query_mask = torch.cat([torch.ones_like(pts_ref[..., 0]).bool(), query_mask_img], dim=1)
            reference_points = torch.cat([pts_ref, reference_points_img], dim=1)

            num_query_img = int(self.num_query - pts_ref.size(1))

        # denoise training
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)

        # mask out padded query for attention
        tgt_size = self.num_query
        src_size = self.num_query
        if attn_mask is None:
            attn_mask = torch.zeros((tgt_size, src_size), dtype=torch.bool, device=reference_points.device)
        pad_size = attn_mask.size(0) - tgt_size
        if mask_dict is not None:
            assert pad_size == mask_dict['pad_size']
        attn_mask = attn_mask.repeat(B, 1, 1)

        tgt_query_mask = query_mask
        src_query_mask = query_mask
        attn_mask[:, :, pad_size:] = ~src_query_mask[:, None]
        num_heads = self.transformer.decoder.layers[0].attentions[0].num_heads
        attn_mask = attn_mask.repeat_interleave(num_heads, dim=0)

        # query content feature
        if self.dynamic_q_zero_init:
            tgt = self.dyn_q_embed.weight.new_zeros((B, num_query_img, self.embed_dims)) if num_query_img > 0 else \
                  torch.empty(B, 0, self.embed_dims, device=self.dyn_q_embed.weight.device)
            pts_tgt = self.pts_q_embed.weight.new_zeros((B, num_query_pts, self.embed_dims)) if num_query_pts > 0 else \
                      torch.empty(B, 0, self.embed_dims, device=self.pts_q_embed.weight.device)
        else:
            tgt = self.dyn_q_embed.weight.repeat(B, num_query_img, 1) if num_query_img > 0 else \
                  torch.empty(B, 0, self.embed_dims, device=self.dyn_q_embed.weight.device)
            pts_tgt = self.pts_q_embed.weight.repeat(B, num_query_pts, 1) if num_query_pts > 0 else \
                      torch.empty(B, 0, self.embed_dims, device=self.pts_q_embed.weight.device)

        tgt = torch.cat([tgt.new_zeros((B, pad_size, self.embed_dims)), pts_tgt, tgt], dim=1)
        
        if not self.dynamic_q_zero_init:
            pad_query_feats = query_feats.new_zeros([B, pad_size + self.num_query, self.embed_dims])
            if num_query_pts > 0:
                pts_query_feat_emb = self.pts_query_embed(pts_query_feat)
                pad_query_feats[:, pad_size:pad_size + num_query_pts] = pts_query_feat_emb
            if num_query_img > 0:
                pad_query_feats[:, pad_size + num_query_pts:pad_size + self.num_query] = query_feats
            tgt = self.dyn_q_enc(tgt, pad_query_feats)

        # query positional encoding
        query_pos = self.query_embedding(pos2posemb3d(reference_points))

        # encode position distribution for image query
        if num_query_img > 0:
            query_pos_det = self.dyn_q_pos(query_coords.flatten(-2, -1))
            query_pos_det = self.dyn_q_pos_with_prob(query_pos_det, query_probs)
            query_pos[:, pad_size + num_query_pts:pad_size + self.num_query] = query_pos_det

        dyn_q_mask = torch.zeros_like(tgt[..., 0]).bool()
        if num_query_img > 0:
            dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query] = 1
        dyn_q_mask[:, pad_size:] &= tgt_query_mask
        
        dyn_q_mask_img = dyn_q_mask[:, pad_size + num_query_pts:pad_size + self.num_query]
        dyn_q_coords = query_coords[dyn_q_mask_img] if num_query_img > 0 else query_coords
        dyn_q_probs = query_probs[dyn_q_mask_img] if num_query_img > 0 else query_probs

        # transformer decoder
        all_reference_points = reference_points.clone()

        outs_dec, reference_points, dyn_q_logits = self.transformer(
            tgt, query_pos, attn_mask,
            feat_flatten_img, spatial_flatten_img, level_start_index_img, self.pc_range, img_metas, data['lidar2img'],
            feat_flatten_pts=feat_flatten_pts, pos_flatten_pts=pos_flatten_pts,
            cross_attn_masks=None, reference_points=reference_points,
            dyn_q_coords=dyn_q_coords, dyn_q_probs=dyn_q_probs, dyn_q_mask=dyn_q_mask, dyn_q_pos_branch=self.dyn_q_pos,
            dyn_q_pos_with_prob_branch=self.dyn_q_pos_with_prob, dyn_q_prob_branch=self.dyn_q_prob_branch,
            save_modal_weights=self.modal_weight,
        )

        # Collect modality weights.
        all_modal_weights_img = None
        all_modal_weights_pts = None
        modal_weight_stats = None
        if self.modal_weight:
            all_modal_weights_img, all_modal_weights_pts = self._collect_modal_weights()
            
            # Compute modality weight statistics.
            if all_modal_weights_img is not None and all_modal_weights_pts is not None:
                modal_weight_stats = self._log_modal_weight_statistics(
                    all_modal_weights_img, all_modal_weights_pts, 
                    pad_size=mask_dict['pad_size'] if mask_dict else 0
                )

        # generate prediction
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points[lvl].clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
                    all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # Clone raw predictions and initial target query mask
        # _all_cls_scores_unmasked contains scores before any NMS or specific modal filtering.
        # Shape: (num_layers, B, pad_size + self.num_query, num_classes)
        _all_cls_scores_unmasked = all_cls_scores.clone()
        # _all_bbox_preds_unmasked contains bbox predictions similarly.
        # Shape: (num_layers, B, pad_size + self.num_query, code_size)
        _all_bbox_preds_unmasked = all_bbox_preds.clone()
        # _all_modal_weights_unmasked contains modal weights if collected
        # Shape: (num_layers, B, pad_size + self.num_query, 1) or None
        _all_modal_weights_img_unmasked = all_modal_weights_img.clone() if all_modal_weights_img is not None else None
        _all_modal_weights_pts_unmasked = all_modal_weights_pts.clone() if all_modal_weights_pts is not None else None
        _all_reference_points_unmasked = all_reference_points.clone()
        # _initial_tgt_query_mask indicates valid queries among the self.num_query (non-padded) part.
        # Shape: (B, self.num_query)
        _initial_tgt_query_mask = tgt_query_mask.clone() 

        outs = {}
        outs['num_query_pts'] = num_query_pts
        
        # Save modal_weight_stats for logging.
        if modal_weight_stats is not None:
            outs['modal_weight_stats'] = modal_weight_stats
        
        # Handle denoise mask dictionary and split predictions
        if mask_dict and mask_dict['pad_size'] > 0:
            current_pad_size = mask_dict['pad_size']
            output_known_class = _all_cls_scores_unmasked[:, :, :current_pad_size, :]
            output_known_coord = _all_bbox_preds_unmasked[:, :, :current_pad_size, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_class, output_known_coord)
            outs['dn_mask_dict'] = mask_dict

            # Predictions for matching/NMS are after the padded part
            cls_scores_for_processing = _all_cls_scores_unmasked[:, :, current_pad_size:, :]
            bbox_preds_for_processing = _all_bbox_preds_unmasked[:, :, current_pad_size:, :]
            # Process modal_weights as well, skipping the padded part.
            modal_weights_img_for_processing = _all_modal_weights_img_unmasked[:, :, current_pad_size:, :] if _all_modal_weights_img_unmasked is not None else None
            modal_weights_pts_for_processing = _all_modal_weights_pts_unmasked[:, :, current_pad_size:, :] if _all_modal_weights_pts_unmasked is not None else None
            reference_points_for_processing = _all_reference_points_unmasked[:, current_pad_size:, :]
            # _initial_tgt_query_mask (shape B, self.num_query) applies to this non-padded part
            active_query_mask_base = _initial_tgt_query_mask
        else:
            outs['dn_mask_dict'] = None
            current_pad_size = 0  # Set default value
            # Process all queries if no DN padding
            cls_scores_for_processing = _all_cls_scores_unmasked
            bbox_preds_for_processing = _all_bbox_preds_unmasked
            modal_weights_img_for_processing = _all_modal_weights_img_unmasked
            modal_weights_pts_for_processing = _all_modal_weights_pts_unmasked
            reference_points_for_processing = _all_reference_points_unmasked
            active_query_mask_base = _initial_tgt_query_mask
        
        num_active_queries = active_query_mask_base.size(1) # This is self.num_query (pts + img_dynamic)
        B = cls_scores_for_processing.size(1) # Batch size

        # NMS parameters
        iou_thr = self.post_bev_nms_thr
        score_thr = self.post_bev_nms_score
        ops = self.post_bev_nms_ops

        if self.training:
            # Training: simple output based on current mode and generated queries
            final_keep_mask = active_query_mask_base.clone()

            if len(ops) > 0 and 0 in ops: # Assuming 0 is the op for this NMS type
                assert len(ops) == 1 # As per original logic
                
                # Prepare NMS inputs from the last decoder layer
                bbox_output_last_layer = denormalize_bbox(bbox_preds_for_processing[-1], None)
                bbox_bev_last_layer = bbox_output_last_layer[..., [0, 1, 3, 4, 6]] 
                score_bev_last_layer = cls_scores_for_processing[-1].sigmoid().max(-1).values

                nms_passed_mask = torch.zeros_like(active_query_mask_base)

                for b_idx in range(B):
                    active_indices_b = torch.where(active_query_mask_base[b_idx])[0]
                    if len(active_indices_b) > 0:
                        boxes_for_nms_b = bbox_bev_last_layer[b_idx, active_indices_b]
                        scores_for_nms_b = score_bev_last_layer[b_idx, active_indices_b]
                        
                        keep_indices_relative = nms_bev(
                            xywhr2xyxyr(boxes_for_nms_b), 
                            scores_for_nms_b, 
                            iou_thr, 
                            pre_max_size=None,
                            post_max_size=None
                        )
                        if len(keep_indices_relative) > 0:
                            absolute_kept_indices = active_indices_b[keep_indices_relative]
                            nms_passed_mask[b_idx, absolute_kept_indices] = True
                
                score_ok_mask = (cls_scores_for_processing[-1].sigmoid().max(-1).values > score_thr)
                final_keep_mask = nms_passed_mask & score_ok_mask & active_query_mask_base

            # Apply final_keep_mask to all layers
            expanded_final_keep_mask = final_keep_mask.unsqueeze(0).unsqueeze(-1)

            masked_cls_scores = torch.where(
                expanded_final_keep_mask, 
                cls_scores_for_processing, 
                torch.full_like(cls_scores_for_processing, -40.)
            )
            masked_bbox_preds = torch.where(
                expanded_final_keep_mask, 
                bbox_preds_for_processing, 
                torch.full_like(bbox_preds_for_processing, 0.)
            )
            outs['all_cls_scores'] = masked_cls_scores
            outs['all_bbox_preds'] = masked_bbox_preds
            outs['all_reference_points'] = reference_points_for_processing
            if modal_weights_img_for_processing is not None and modal_weights_pts_for_processing is not None:
                outs['all_modal_weights_img'] = modal_weights_img_for_processing
                outs['all_modal_weights_pts'] = modal_weights_pts_for_processing
        else:
            # Inference: only compute the required modal output for efficiency
            mode_name = mode if mode in ['pts', 'img', 'all'] else 'all'
            
            # Apply modal filtering to the predictions themselves
            modal_filter = torch.zeros_like(active_query_mask_base) # Shape: (B, num_active_queries)
            if mode_name == 'pts':
                modal_filter[:, :num_query_pts] = True
            elif mode_name == 'img':
                modal_filter[:, num_query_pts:num_active_queries] = True
            else: # 'all'
                modal_filter[:, :num_active_queries] = True
            
            # Apply modal filter to mask out irrelevant queries
            modal_mask_expanded = modal_filter.unsqueeze(0).unsqueeze(-1)  # [1, B, num_queries, 1]
            
            current_cls_scores_loop = torch.where(
                modal_mask_expanded, 
                cls_scores_for_processing, 
                torch.full_like(cls_scores_for_processing, -40.)
            )
            current_bbox_preds_loop = torch.where(
                modal_mask_expanded, 
                bbox_preds_for_processing, 
                torch.full_like(bbox_preds_for_processing, 0.)
            )

            
            active_mask_this_mode_before_nms = active_query_mask_base & modal_filter
            final_keep_mask = active_mask_this_mode_before_nms.clone()

            if len(ops) > 0 and 0 in ops: # Assuming 0 is the op for this NMS type
                assert len(ops) == 1 # As per original logic
                
                # Prepare NMS inputs from the last decoder layer
                # Shape: (B, num_active_queries, D)
                bbox_output_last_layer = denormalize_bbox(current_bbox_preds_loop[-1], None)
                # Shape: (B, num_active_queries, 5) for bev nms (x,y,w,l,angle)
                bbox_bev_last_layer = bbox_output_last_layer[..., [0, 1, 3, 4, 6]] 
                # Shape: (B, num_active_queries)
                score_bev_last_layer = current_cls_scores_loop[-1].sigmoid().max(-1).values

                nms_passed_mask = torch.zeros_like(active_mask_this_mode_before_nms) # (B, num_active_queries)

                for b_idx in range(B):
                    # Consider only queries active for this mode before NMS
                    # active_indices_b are indices within num_active_queries dimension
                    active_indices_b = torch.where(active_mask_this_mode_before_nms[b_idx])[0]
                    if len(active_indices_b) > 0:
                        boxes_for_nms_b = bbox_bev_last_layer[b_idx, active_indices_b]
                        scores_for_nms_b = score_bev_last_layer[b_idx, active_indices_b]
                        
                        keep_indices_relative = nms_bev(
                            xywhr2xyxyr(boxes_for_nms_b), 
                            scores_for_nms_b, 
                            iou_thr, 
                            pre_max_size=None, # Retain original behavior or make configurable
                            post_max_size=None
                        )
                        if len(keep_indices_relative) > 0:
                            absolute_kept_indices = active_indices_b[keep_indices_relative]
                            nms_passed_mask[b_idx, absolute_kept_indices] = True
                
                score_ok_mask = (current_cls_scores_loop[-1].sigmoid().max(-1).values > score_thr)
                final_keep_mask = nms_passed_mask & score_ok_mask & active_mask_this_mode_before_nms
            
            # Apply final_keep_mask to all layers for this mode
            expanded_final_keep_mask = final_keep_mask.unsqueeze(0).unsqueeze(-1)

            masked_cls_scores = torch.where(
                expanded_final_keep_mask, 
                current_cls_scores_loop, 
                torch.full_like(current_cls_scores_loop, -40.)
            )
            masked_bbox_preds = torch.where(
                expanded_final_keep_mask, 
                current_bbox_preds_loop, 
                torch.full_like(current_bbox_preds_loop, 0.)
            )
            # Store outputs based on the requested mode
            if mode_name == 'all':
                outs['all_cls_scores'] = masked_cls_scores
                outs['all_bbox_preds'] = masked_bbox_preds
                outs['all_reference_points'] = reference_points_for_processing
                if modal_weights_img_for_processing is not None and modal_weights_pts_for_processing is not None:
                    outs['all_modal_weights_img'] = modal_weights_img_for_processing
                    outs['all_modal_weights_pts'] = modal_weights_pts_for_processing
            elif mode_name == 'pts':
                outs['pts_cls_scores'] = masked_cls_scores
                outs['pts_bbox_preds'] = masked_bbox_preds
                outs['pts_reference_points'] = reference_points_for_processing
            elif mode_name == 'img':
                outs['img_cls_scores'] = masked_cls_scores
                outs['img_bbox_preds'] = masked_bbox_preds
                outs['img_reference_points'] = reference_points_for_processing

        return outs

    def _collect_modal_weights(self):
        """
        Collect modality weights from each transformer decoder layer.
        
        Returns:
            tuple: (all_weights_img, all_weights_pts) or (None, None)
                - all_weights_img: [num_layers, B, num_queries, 1], image modality weights.
                - all_weights_pts: [num_layers, B, num_queries, 1], point cloud modality weights.
        """
        if not hasattr(self, 'transformer'):
            return None, None
            
        if not hasattr(self.transformer, 'decoder'):
            return None, None
        
        all_weights_img = []
        all_weights_pts = []
        for layer_idx, layer in enumerate(self.transformer.decoder.layers):
            found_in_layer = False
            for attn_idx, attn in enumerate(layer.attentions):
                if hasattr(attn, '_last_modal_weight_img') and hasattr(attn, '_last_modal_weight_pts'):
                    weights_img = attn._last_modal_weight_img
                    weights_pts = attn._last_modal_weight_pts
                    all_weights_img.append(weights_img)
                    all_weights_pts.append(weights_pts)
                    found_in_layer = True
                    break
            
            if not found_in_layer:
                return None, None
        
        if len(all_weights_img) == 0:
            return None, None
        
        # Stack into [num_layers, B, num_queries, 1].
        return torch.stack(all_weights_img, dim=0), torch.stack(all_weights_pts, dim=0)

    def _log_modal_weight_statistics(self, all_modal_weights_img, all_modal_weights_pts, pad_size=0):
        """
        Compute modality weight statistics (mean and var) for loss logging.
        
        Args:
            all_modal_weights_img: [num_layers, B, num_queries, 1], image modality weights.
            all_modal_weights_pts: [num_layers, B, num_queries, 1], point cloud modality weights.
            pad_size: padding size for denoising queries.
            
        Returns:
            dict: Statistics dictionary with detached tensors for logging.
        """
        stats = {}
        
        # Remove the padded part and only count valid queries.
        if pad_size > 0:
            weights_img = all_modal_weights_img[:, :, pad_size:, :]
            weights_pts = all_modal_weights_pts[:, :, pad_size:, :]
        else:
            weights_img = all_modal_weights_img
            weights_pts = all_modal_weights_pts
        
        # Flatten weights across all layers and batches [num_layers * B * num_queries, 1].
        weights_img_flat = weights_img.reshape(-1)
        weights_pts_flat = weights_pts.reshape(-1)
        
        # Compute global statistics.
        stats['modal_weight_img_mean'] = weights_img_flat.mean().detach()
        stats['modal_weight_img_var'] = weights_img_flat.var().detach()
        stats['modal_weight_img_std'] = weights_img_flat.std().detach()
        
        stats['modal_weight_pts_mean'] = weights_pts_flat.mean().detach()
        stats['modal_weight_pts_var'] = weights_pts_flat.var().detach()
        stats['modal_weight_pts_std'] = weights_pts_flat.std().detach()
        
        return stats

    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None,
                           branch='all'):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
            branch (str): Branch type ('all', 'pts', 'img').
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
                - ious (Tensor or None): IoU values for each query, None for HungarianAssigner3D (no IoU recording).
        """

        num_bboxes = bbox_pred.size(0)
        
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # Extract IoU information from assign_result
        # Support both HungarianAssigner3D and HungarianAssigner3DV3 outputs.
        if assign_result.max_overlaps is not None:
            ious = assign_result.max_overlaps
        else:
            # HungarianAssigner3D returns None; keep None to indicate IoU should not be logged.
            ious = None

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        if sampling_result.num_gts > 0 and pos_inds.numel() > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds, ious)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    branch='all'):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
            branch (str): Branch type ('all', 'pts', 'img').
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list, ious_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,
            gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list, [branch] * num_imgs)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg, pos_inds_list, ious_list)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds',))
    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None,
                    branch='all'):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
            branch (str): Branch type ('all', 'pts', 'img').
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # num_imgs = cls_scores.size(0)
        num_imgs = len(cls_scores)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list, branch)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg, pos_inds_list, ious_list) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        if isinstance(cls_scores, (tuple, list)):
            cls_scores = torch.cat(cls_scores, dim=0)
        else:
            cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        if len(cls_scores) == 0:
            loss_cls = cls_scores.sum() * cls_avg_factor
        else:
            loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        if isinstance(bbox_preds, (tuple, list)):
            bbox_preds = torch.cat(bbox_preds, dim=0)
        else:
            bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox, pos_inds_list, ious_list

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split * self.split  ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels).float()
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None,
             log_stats=False,
             branch='all'):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
            log_stats (bool): Whether to log statistics about query assignment.
            branch (str): Branch type ('all', 'pts', 'img').
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # --- GT Filtering based on visibility mask ---
        if branch in ['pts', 'img']:
            filtered_gt_bboxes_list = []
            filtered_gt_labels_list = []
            
            visibility_key = f'gt_{branch}_visible_mask'

            for i in range(len(gt_bboxes_list)):
                gt_bboxes = gt_bboxes_list[i]  # Stays on CPU
                gt_labels = gt_labels_list[i]  # Is on GPU
                
                # Default mask is all True, created on CPU
                visible_mask_cpu = torch.ones(len(gt_labels), dtype=torch.bool, device='cpu')

                if img_metas is not None and visibility_key in img_metas[i]:
                    mask_from_meta = img_metas[i][visibility_key]
                    if len(mask_from_meta) == len(gt_labels):
                        visible_mask_cpu = mask_from_meta.to('cpu') # Ensure it's a CPU tensor
                
                # Use CPU mask for CPU tensor (gt_bboxes)
                filtered_gt_bboxes_list.append(gt_bboxes[visible_mask_cpu])
                
                # Use GPU mask for GPU tensor (gt_labels)
                visible_mask_gpu = visible_mask_cpu.to(gt_labels.device)
                filtered_gt_labels_list.append(gt_labels[visible_mask_gpu])
            
            # Replace original lists with filtered lists for loss calculation
            gt_bboxes_list = filtered_gt_bboxes_list
            gt_labels_list = filtered_gt_labels_list
        # --- End of GT Filtering ---

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        self.assigner.layer_indicator = 1
        losses_cls, losses_bbox, pos_inds_list_per_layer, ious_list_per_layer = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list,
            all_gt_bboxes_ignore_list, [branch] * num_dec_layers)
        self.assigner.layer_indicator = 0

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        
        # log matched query source
        if log_stats:
            num_query_pts = preds_dicts['num_query_pts']
            pos_inds_last_layer = pos_inds_list_per_layer[-1]
            
            num_matched_pts = 0
            num_matched_img = 0
            
            # pos_inds_last_layer is a list of tensors, one for each sample in the batch
            for pos_inds in pos_inds_last_layer:
                if num_query_pts > 0:
                    num_matched_pts += (pos_inds < num_query_pts).sum().item()
                    num_matched_img += (pos_inds >= num_query_pts).sum().item()
            
            loss_dict['num_matched_pts'] = losses_cls[-1].new_tensor(num_matched_pts)
            loss_dict['num_matched_img'] = losses_cls[-1].new_tensor(num_matched_img)

        # Record matched IoUs from assigner results
        ious_last_layer = ious_list_per_layer[-1]
        matched_ious = []
        
        # Check whether any sample contains valid IoU information.
        has_valid_ious = False
        for ious in ious_last_layer:
            if ious is not None:
                has_valid_ious = True
                # Get positive-sample IoUs. IoU > 0 means a matched positive sample.
                pos_mask = ious > 0
                if pos_mask.sum() > 0:
                    pos_ious = ious[pos_mask]
                    matched_ious.extend(pos_ious.detach().cpu().numpy().tolist())
        
        # Only log to loss_dict when valid IoU information exists.
        if has_valid_ious and matched_ious:
            # Compute average IoU.
            avg_iou = sum(matched_ious) / len(matched_ious)
            # Add a branch-specific key suffix.
            loss_dict[f'matched_ious_{branch}'] = losses_cls[-1].new_tensor(avg_iou)
        # If there is no valid IoU information, such as with HungarianAssigner3D, do not log matched_ious.
        else:
            loss_dict[f'matched_ious_{branch}'] = losses_cls[-1].new_tensor(0.0)

        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]

            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list,
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1

        elif self.with_dn:
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_gt_bboxes_list, all_gt_labels_list,
                all_gt_bboxes_ignore_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                               dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()
                num_dec_layer += 1

        if 'modal_weight_stats' in preds_dicts:
            modal_weight_stats = preds_dicts['modal_weight_stats']
            if modal_weight_stats is not None:
                for key, value in modal_weight_stats.items():
                    loss_dict[key] = value

        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # Determine which modal outputs to use based on what's available
        # Priority: specific modal outputs > all outputs
        if 'img_cls_scores' in preds_dicts and 'pts_cls_scores' not in preds_dicts:
            # Use image modal outputs
            preds_dicts['all_cls_scores'] = preds_dicts['img_cls_scores']
            preds_dicts['all_bbox_preds'] = preds_dicts['img_bbox_preds']
        elif 'pts_cls_scores' in preds_dicts and 'img_cls_scores' not in preds_dicts:
            # Use point cloud modal outputs
            preds_dicts['all_cls_scores'] = preds_dicts['pts_cls_scores']
            preds_dicts['all_bbox_preds'] = preds_dicts['pts_bbox_preds']
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            
            ret_dict = dict(
                boxes_3d=img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1)),
                scores_3d=preds['scores'],
                labels_3d=preds['labels'],
            )

            for field in self.save_fields:
                if field in preds:
                    ret_dict[field] = preds[field]
                
            ret_list.append(ret_dict)
        return ret_list

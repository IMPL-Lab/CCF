# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.core import bbox2roi
from mmdet.models.utils import build_linear_layer

from ..builder import QUERY_GENERATORS
from .image_singple_point_query_generator import ImageSinglePointQueryGenerator

@QUERY_GENERATORS.register_module()
class ImageDistributionQueryGenerator(ImageSinglePointQueryGenerator):
    def __init__(self,
                 prob_bin=50,
                 depth_range=[0.1, 90],
                 x_range=[-10, 10],
                 y_range=[-8, 8],
                 code_size=10,

                 gt_guided_loss=1.0,
                 gt_guided=False,
                 gt_guided_weight=0.2,
                 gt_guided_ratio=0.075,
                 gt_guided_prob=0.5,
                 
                 # LiDAR prior fusion parameters
                 use_lidar_prior=False,
                 fusion_type='add',  # 'add', 'product', or 'concat'
                 weight_type='learnable',  # 'learnable' or 'adaptive'
                 fusion_weight=0.0,  # initial weight (only used when weight_type='learnable', 0.0 because of sigmoid)
                 
                 # Adaptive weight network parameters
                 confidence_hidden_dim=64,
                 depth_loss_type='ce',
                 depth_detach=False,
                 **kwargs,
                 ):
        super(ImageDistributionQueryGenerator, self).__init__(**kwargs)
        self.prob_bin = prob_bin
        center = np.array(self.roi_feat_size)
        x_bins = np.linspace(center[0] + x_range[0], center[0] + x_range[1], prob_bin)
        y_bins = np.linspace(center[1] + y_range[0], center[1] + y_range[1], prob_bin)
        d_bins = np.linspace(depth_range[0], depth_range[1], prob_bin)
        xyd_bins = torch.tensor(np.stack([x_bins, y_bins, d_bins], axis=-1), dtype=torch.float)
        self.register_buffer('xyd_bins', xyd_bins)

        self.gt_guided = gt_guided
        self.gt_guided_weight = gt_guided_weight
        self.gt_guided_ratio = gt_guided_ratio
        self.gt_guided_prob = gt_guided_prob
        self.gt_guided_loss = gt_guided_loss

        self.code_size = code_size

        self.depth_loss_type = depth_loss_type
        self.depth_detach = depth_detach
        
        # LiDAR prior fusion related parameters
        self.use_lidar_prior = use_lidar_prior
        self.fusion_type = fusion_type
        self.weight_type = weight_type
        
        if self.use_lidar_prior:
            if fusion_type == 'concat':
                self.fusion_mlp = nn.Sequential(
                    nn.Linear(prob_bin * 2, confidence_hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(confidence_hidden_dim, confidence_hidden_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(confidence_hidden_dim // 2, prob_bin),
                )
            else:
                # weight calculation method (for add and product fusion)
                if weight_type == 'learnable':
                    # learnable fusion weight
                    self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))
                elif weight_type == 'adaptive':
                    input_dim = prob_bin * 2  # image depth probability + LiDAR depth distribution
                    
                    # LayerNorm for normalizing all input features
                    self.confidence_input_norm = nn.LayerNorm(input_dim)
                    
                    # adaptive weight network
                    self.confidence_net = nn.Sequential(
                        nn.Linear(input_dim, confidence_hidden_dim),  # image depth probability + LiDAR depth distribution
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.1),
                        nn.Linear(confidence_hidden_dim, confidence_hidden_dim // 2),
                        nn.ReLU(inplace=True),
                        nn.Linear(confidence_hidden_dim // 2, 1),  # output single weight value
                        nn.Sigmoid()  # ensure output is in [0,1] range
                    )
                else:
                    raise ValueError(f"Unsupported weight_type: {weight_type}")

        self.fc_center = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.center_last_dim,
                out_features=prob_bin * 3)

        self.batch_split = True
        
    def process_lidar_prior(self, roi_lidar_depth):
        """
        Convert LiDAR depth values within ROI to depth distribution
        
        Args:
            roi_lidar_depth: List[Tensor], LiDAR depth values within each ROI
        
        Returns:
            depth_distribution: Tensor [n_rois, prob_bin]
        """
        depth_bins = self.xyd_bins[:, 2]  # [prob_bin]
        n_rois = len(roi_lidar_depth)
        n_bins = len(depth_bins)
        
        # pre-allocated result tensor
        depth_distributions = torch.zeros(n_rois, n_bins, device=depth_bins.device)
        
        # calculate the range and step of bins
        bin_min = depth_bins[0]
        bin_max = depth_bins[-1]
        bin_range = bin_max - bin_min
        
        for i, depths in enumerate(roi_lidar_depth):
            if len(depths) == 0:
                # use uniform distribution when there are no LiDAR points
                depth_distributions[i] = 1.0 / n_bins
            else:
                # vectorized linear interpolation
                # normalize depth values to [0, n_bins-1] range
                depths_normalized = (depths - bin_min) / bin_range * (n_bins - 1)
                depths_normalized = torch.clamp(depths_normalized, 0, n_bins - 1)
                
                # calculate left and right indices
                left_indices = torch.floor(depths_normalized).long()
                right_indices = torch.clamp(left_indices + 1, 0, n_bins - 1)
                
                # calculate interpolation weights
                weights_right = depths_normalized - left_indices.float()
                weights_left = 1.0 - weights_right
                
                # use scatter_add for efficient weight allocation
                hist = torch.zeros(n_bins, device=depths.device)
                hist.scatter_add_(0, left_indices, weights_left)
                
                # for right_indices, only add when it is different from left_indices
                valid_right_mask = right_indices != left_indices
                if valid_right_mask.any():
                    hist.scatter_add_(0, right_indices[valid_right_mask], weights_right[valid_right_mask])
                
                # normalize to probability distribution
                hist_sum = hist.sum()
                depth_distributions[i] = hist / (hist_sum + 1e-8) if hist_sum > 0 else torch.ones_like(hist) / n_bins
        
        return depth_distributions

    def fuse_depth_distributions(self, image_depth_logits, lidar_depth_dist):
        """
        Fuse image predicted depth distribution and LiDAR prior distribution
        
        Args:
            image_depth_logits: Tensor [n_rois, prob_bin], image predicted logits
            lidar_depth_dist: Tensor [n_rois, prob_bin], LiDAR prior distribution
        
        Returns:
            fused_prob: Tensor [n_rois, prob_bin], fused probability distribution
            fusion_info: Dict, fusion related information (for debugging and visualization)
        """
        image_depth_prob = F.softmax(image_depth_logits, dim=-1)
        fusion_info = {}

        if self.fusion_type == 'concat':
            # concat fusion: directly concatenate two probability distributions and then pass through MLP to get fused result
            concat_feat = torch.cat([image_depth_prob, lidar_depth_dist], dim=-1)  # [n_rois, prob_bin*2]
            fused_logits = self.fusion_mlp(concat_feat)  # [n_rois, prob_bin]
            # clip MLP output logits
            fused_logits = torch.clamp(fused_logits, min=-10, max=10)
            fused_prob = F.softmax(fused_logits, dim=-1)
            
            fusion_info['fusion_type'] = 'concat'
            fusion_info['concat_entropy'] = -(fused_prob * torch.log(fused_prob + 1e-8)).sum(-1).mean().item()
        
        else:
            if self.weight_type == 'learnable':
                # use learnable scalar weight
                alpha = torch.sigmoid(self.fusion_weight)  # ensure output is in [0,1] range
                image_weight = alpha
                lidar_weight = 1 - alpha
                fusion_info['fusion_weight'] = alpha.item()
                
            elif self.weight_type == 'adaptive':
                # Use an adaptive weight network over image and LiDAR depth distributions.
                concat_feat = torch.cat([image_depth_prob, lidar_depth_dist], dim=-1)  # [n_rois, prob_bin*2]
                concat_feat = self.confidence_input_norm(concat_feat)

                confidence_weights = self.confidence_net(concat_feat)  # [n_rois, 1]
                image_weight = confidence_weights
                lidar_weight = 1 - confidence_weights

                fusion_info['fusion_weight'] = image_weight.mean().item()

            else:
                raise ValueError(f"Unsupported weight_type: {self.weight_type}")
            
            if self.fusion_type == 'add':
                # weighted addition fusion (weight sum is 1, result is naturally probability distribution)
                fused_prob = image_weight * image_depth_prob + lidar_weight * lidar_depth_dist
                assert ((fused_prob.sum(-1) - 1).abs() < 1e-6).all(), "Fused probability distribution is incorrect"
                
            elif self.fusion_type == 'product':
                # weighted probability product fusion (log space operation, need softmax normalization)
                # use clamp to prevent values close to 0
                image_depth_prob_safe = torch.clamp(image_depth_prob, min=1e-6, max=1.0 - 1e-6)
                lidar_depth_dist_safe = torch.clamp(lidar_depth_dist, min=1e-6, max=1.0 - 1e-6)
                
                log_image_prob = torch.log(image_depth_prob_safe)
                log_lidar_prob = torch.log(lidar_depth_dist_safe)
                
                # clip log values to prevent extreme values
                log_image_prob = torch.clamp(log_image_prob, min=-10, max=0)
                log_lidar_prob = torch.clamp(log_lidar_prob, min=-10, max=0)
                
                # weighted fusion in log space
                weighted_log_sum = image_weight * log_image_prob + lidar_weight * log_lidar_prob
                # clip again to ensure stable softmax input
                weighted_log_sum = torch.clamp(weighted_log_sum, min=-10, max=10)
                fused_prob = F.softmax(weighted_log_sum, dim=-1)
            
            else:
                raise ValueError(f"Unsupported fusion_type: {self.fusion_type}")
            
        return fused_prob, fusion_info
    
    @auto_fp16(apply_to=('x', ))
    def forward(self, x, proposal_list, img_metas, **kwargs):
        n_rois_per_view, n_rois_per_batch = kwargs['n_rois_per_view'], kwargs['n_rois_per_batch']

        intrinsics, extrinsics = self.get_box_params(proposal_list,
                                                     [img_meta['intrinsics'] for img_meta in img_metas],
                                                     [img_meta['extrinsics'] for img_meta in img_metas])
        extra_feats = dict(intrinsic=self.process_intrins_feat(intrinsics))

        qg_args = dict()
        qg_args['rois'] = bbox2roi(proposal_list)
        if 'sparse_depth' in kwargs['data']:
            qg_args['sparse_depth'] = kwargs['data']['sparse_depth'][:, :, 2]
            qg_args['img'] = kwargs['data']['img']

        if self.training:
            qg_args['gt_bboxes'] = kwargs['data']['gt_bboxes']
            qg_args['gt_depths'] = kwargs['data']['depths']

        roi_feat, return_feats = self.get_roi_feat(x, proposal_list, extra_feats)
        center_pred, return_feats = self.get_prediction(roi_feat, intrinsics, extrinsics, extra_feats, return_feats, n_rois_per_batch, qg_args)

        return center_pred, return_feats

    def get_roi_feat(self, x, proposal_list, extra_feats=dict()):
        roi_feat = x
        return_feats = dict()
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)
        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.flatten(1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))

        # extra encoding
        enc_feat = [x]
        for enc in self.extra_encoding['features']:
            enc_feat.append(extra_feats.get(enc['type']))
        enc_feat = torch.cat(enc_feat, dim=1).clamp(min=-5e3, max=5e3)
        x = self.extra_enc(enc_feat)
        if self.return_cfg.get('enc', False):
            return_feats['enc'] = x

        return x, return_feats

    def get_prediction(self, x, intrinsics, extrinsics, extra_feats, return_feats, n_rois_per_batch, kwargs=dict()):
        x = torch.nan_to_num(x)
        # separate branches
        x_cls = x
        x_center = x
        x_size = x
        x_heading = x
        x_attr = x

        out_dict = {}
        for output in ['cls', 'size', 'heading', 'center', 'attr']:
            out_dict[f'x_{output}'] = self.get_output(eval(f'x_{output}'), getattr(self, f'{output}_convs'),
                                                      getattr(self, f'{output}_fcs'))
            if self.return_cfg.get(output, False):
                return_feats[output] = out_dict[f'x_{output}']

        x_cls = out_dict['x_cls']
        x_center = out_dict['x_center']
        x_size = out_dict['x_size']
        x_heading = out_dict['x_heading']
        x_attr = out_dict['x_attr']

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        size_pred = self.fc_size(x_size) if self.with_size else None
        heading_pred = self.fc_heading(x_heading) if self.with_heading else None
        center_pred = self.fc_center(x_center) if self.with_center else None
        attr_pred = self.fc_attr(x_attr) if self.with_attr else None

        n_rois = center_pred.size(0)
        center_pred = center_pred.view(n_rois, self.prob_bin, 3)
        depth_logits = center_pred[..., 2]
        
        # clip initial depth_logits to prevent network output extreme values
        depth_logits = torch.clamp(depth_logits, min=-10, max=10)
        
        # LiDAR prior fusion
        if self.use_lidar_prior and 'sparse_depth' in kwargs:
            
            # import time
            # start_time = time.time()

            sparse_depth = kwargs['sparse_depth']
            sparse_depth_scale = sparse_depth.shape[-2:]
            sparse_depth = sparse_depth.view(-1, sparse_depth_scale[0], sparse_depth_scale[1])
            img_scale = kwargs['img'].shape[-2:]
            scale_factor_w = img_scale[0] / sparse_depth_scale[0]
            scale_factor_h = img_scale[1] / sparse_depth_scale[1]
            assert scale_factor_w == scale_factor_h, "Scale factor of width and height must be the same"

            # generate LiDAR depth points for each ROI
            roi_lidar_depth = []
            rois = kwargs['rois']  # [N, 5] format: [batch_ind, x1, y1, x2, y2]
            
            for roi in rois:
                batch_ind = int(roi[0])
                x1, y1, x2, y2 = roi[1:5]
                
                # convert ROI coordinates from img scale to sparse_depth scale
                x1_scaled = x1 / scale_factor_w
                y1_scaled = y1 / scale_factor_h
                x2_scaled = x2 / scale_factor_w
                y2_scaled = y2 / scale_factor_h
                
                # convert to integer coordinates and ensure within valid range
                x1_int = max(0, int(x1_scaled))
                y1_int = max(0, int(y1_scaled))
                x2_int = min(sparse_depth_scale[1], int(x2_scaled) + 1)  # +1 for inclusive
                y2_int = min(sparse_depth_scale[0], int(y2_scaled) + 1)
                
                # extract sparse depth within ROI region
                roi_sparse_depth = sparse_depth[batch_ind, y1_int:y2_int, x1_int:x2_int]
                
                # get valid depth values (>0)
                valid_mask = roi_sparse_depth > 0
                valid_depths = roi_sparse_depth[valid_mask]
                
                # directly use valid depth values
                roi_lidar_depth.append(valid_depths)

            # end_time = time.time()
            # print(f"Prepare LiDAR prior time: {end_time - start_time}s")

            # start_time = time.time()
            lidar_depth_dist = self.process_lidar_prior(roi_lidar_depth)
            # end_time = time.time()
            # print(f"Process LiDAR prior time: {end_time - start_time}s")

            # start_time = time.time()
            depth_prob, fusion_info = self.fuse_depth_distributions(depth_logits, lidar_depth_dist)
            # end_time = time.time()
            # print(f"Fuse depth distributions time: {end_time - start_time}s")

            return_feats['lidar_depth_prior'] = lidar_depth_dist
            return_feats.update(fusion_info)
        else:
            depth_prob = F.softmax(depth_logits, dim=-1)

        depth_integral = torch.matmul(depth_prob, self.xyd_bins[:, 2:3])
        xy_integral = torch.matmul(depth_prob[:, None], center_pred[..., 0:2])[:, 0]

        center_integral = torch.cat([xy_integral, depth_integral], dim=-1)

        if self.with_cls and self.with_size:
            center_lidar = self.center2lidar(center_integral, intrinsics, extrinsics)
            bbox_lidar = torch.cat([center_lidar, size_pred,  center_lidar.new_zeros((center_lidar.size(0), 4))], dim=-1)
            # sin, cos
            bbox_lidar[:, 7] = 1
            return_feats['cls_scores'] = cls_score[:, :self.cls_out_channels]
            return_feats['bbox_preds'] = bbox_lidar[:, :self.code_size]

        assert ((depth_prob.sum(-1) - 1).abs() < 1e-4).all()
        # use gt centers to guide image query generation
        if self.training and (self.gt_guided or self.gt_guided_loss > 0):
            # if LiDAR prior fusion is used, depth_prob should be calculated based on fused result
            if self.use_lidar_prior and 'sparse_depth' in kwargs:
                # convert fused probability distribution to logits for loss calculation
                # use larger epsilon to avoid numerical instability and clip result
                effective_depth_logits = torch.log(depth_prob.clamp(min=1e-6, max=1.0 - 1e-6))
                # further clip logits to prevent extreme values
                effective_depth_logits = effective_depth_logits.clamp(min=-10, max=10)
            else:
                effective_depth_logits = depth_logits

            if self.depth_loss_type == 'ce':
                loss_tag = 'd_loss'
                loss, pred_inds, correct_probs = self.depth_loss(kwargs['gt_bboxes'], kwargs['gt_depths'], kwargs['rois'],
                                                                 effective_depth_logits, self.xyd_bins[:, 2])
            elif self.depth_loss_type == 'silog':
                loss_tag = 'd_silog_loss'
                loss, pred_inds, pred_depths = self.depth_silog_loss(kwargs['gt_bboxes'], kwargs['gt_depths'], kwargs['rois'],
                                                                    effective_depth_logits, self.xyd_bins[:, 2])
            else:
                raise ValueError(f"Unsupported depth_loss_type: {self.depth_loss_type}")
            if loss is not None:
                return_feats[loss_tag] = loss * self.gt_guided_loss
                depth_prob = depth_prob.clone()
                if self.gt_guided:
                    pred_probs = depth_prob[pred_inds]
                    pred_depths = (pred_probs @ self.xyd_bins[:, 2:3])[:, 0]
                    correct_depths = (correct_probs @ self.xyd_bins[:, 2:3])[:, 0]
                    ratio_mask = (pred_depths - correct_depths).abs() > correct_depths * self.gt_guided_ratio
                    prob_mask = torch.rand(pred_inds.shape) <= self.gt_guided_prob
                    mask = ratio_mask & prob_mask.to(ratio_mask.device)
                    depth_prob[pred_inds[mask]] = (1 - self.gt_guided_weight) * pred_probs[mask] + self.gt_guided_weight * correct_probs[mask]
            else:
                return_feats[loss_tag] = x.sum() * 0

            assert ((depth_prob.sum(-1) - 1).abs() < 1e-4).all()

        if self.depth_detach:
            depth_prob_for_decoder = depth_prob.detach()
            center_pred_for_decoder = center_pred.detach()
        else:
            depth_prob_for_decoder = depth_prob
            center_pred_for_decoder = center_pred

        depth_sample = self.xyd_bins[None, :, 2].repeat(n_rois, 1)[..., None]   # [n_rois, prob_bin, 1]
        center_sample = torch.cat([center_pred_for_decoder[..., :2], depth_sample], dim=-1)
        total_size = center_sample.size(1)
        center_sample = self.center2lidar(center_sample.flatten(0, 1),
                                          intrinsics.repeat_interleave(total_size, dim=0),
                                          extrinsics.repeat_interleave(total_size, dim=0))
        center_sample = center_sample.view(n_rois, total_size, 3)
        center_sample = torch.cat([center_sample, depth_prob_for_decoder[..., None]], dim=-1)

        center_sample_batch = center_sample.split(n_rois_per_batch, dim=0)

        return_feats['depth_logits'] = depth_logits
        return_feats['depth_bin'] = self.xyd_bins[:, 2]
        return_feats['query_feats'] = x.split(n_rois_per_batch, dim=0)
        return center_sample_batch, return_feats

    @force_fp32()
    def depth_loss(self, gt_bboxes, gt_depths, rois, depth_logits, depth_bin):
        gt_bboxes = sum(gt_bboxes, [])
        gt_depths = sum(gt_depths, [])
        centers = []
        for img_id, (box, d) in enumerate(zip(gt_bboxes, gt_depths)):
            if box.size(0) > 0:
                img_inds = box.new_full((box.size(0), 1), img_id)
                ct = torch.cat([img_inds, box, d[:, None]], dim=-1)
            else:
                ct = box.new_zeros((0, 6))
            centers.append(ct)
        centers = torch.cat(centers, 0)
        ious = self.box_iou(rois[:, 1:5], centers[:, 1:5])
        if ious.numel() > 0:
            in_same_img = rois[:, 0:1] == centers[None, :, 0]
            ious[~in_same_img] = 0
            new_ious = ious.clone()
            iou_thr = 0.6
            new_ious[ious < iou_thr] = 0

            # 1 pred can only be matched with up to 1 gt
            max_ious_for_preds = new_ious.max(dim=1, keepdim=True).values
            new_ious[new_ious < max_ious_for_preds] = 0
            # 1 gt can only be matched with up to 1 pred
            max_ious_for_gts = new_ious.max(dim=0, keepdim=True).values
            new_ious[new_ious < max_ious_for_gts] = 0
            inds = (new_ious >= iou_thr).nonzero()
            pred_inds, gt_inds = inds[..., 0], inds[..., 1]

            # transform depth values to depth bins
            depth_logits_preds = depth_logits[pred_inds]
            depth_gts = centers[:, 5][gt_inds]
            depth_bin_gts = (depth_gts - depth_bin[0]) / (depth_bin[-1] - depth_bin[0]) * (len(depth_bin) - 1)
            depth_bin_gts = depth_bin_gts.clamp(min=0 + 1e-3, max=len(depth_bin) - 1 - 1e-3)
            depth_bin_gts_l = depth_bin_gts.floor().long()
            depth_bin_gts_r = depth_bin_gts_l + 1
            depth_bin_gts_l_w = depth_bin_gts_r - depth_bin_gts
            depth_bin_gts_r_w = 1 - depth_bin_gts_l_w

            # ce loss for depth bins
            ce_loss_l = F.cross_entropy(depth_logits_preds, depth_bin_gts_l, reduction='none') * depth_bin_gts_l_w
            ce_loss_r = F.cross_entropy(depth_logits_preds, depth_bin_gts_r, reduction='none') * depth_bin_gts_r_w
            ce_loss = (ce_loss_l + ce_loss_r).mean()

            loss = ce_loss
            correct_probs = torch.zeros_like(depth_logits_preds)
            correct_probs.scatter_(1, depth_bin_gts_l[:, None], depth_bin_gts_l_w[:, None])
            correct_probs.scatter_(1, depth_bin_gts_r[:, None], depth_bin_gts_r_w[:, None])
            pred_depths = (correct_probs @ depth_bin[:, None])[:, 0]
            assert (correct_probs.sum(-1) == 1).all()
            assert (((pred_depths - depth_gts).abs() < 2e-1) | (depth_gts > depth_bin[-1])).all()
        else:
            loss = None
            pred_inds = None
            correct_probs = None

        return loss, pred_inds, correct_probs
    
    @force_fp32()
    def depth_silog_loss(self, gt_bboxes, gt_depths, rois, depth_logits, depth_bin):
        """
        Scale-Invariant Logarithmic Loss for depth estimation.
        Reference implementation from AdaBins paper.
        
        Args:
            gt_bboxes: List[Tensor], GT bounding boxes for each image
            gt_depths: List[Tensor], GT depths for each image
            rois: Tensor [n_rois, 5], ROI coordinates [batch_ind, x1, y1, x2, y2]
            depth_logits: Tensor [n_rois, prob_bin], depth predicted logits
            depth_bin: Tensor [prob_bin], depth bin values
        
        Returns:
            loss: SILog loss value, None if no match
            pred_inds: matched prediction indices
            pred_depths: predicted depth values
        """
        gt_bboxes = sum(gt_bboxes, [])
        gt_depths = sum(gt_depths, [])
        centers = []
        for img_id, (box, d) in enumerate(zip(gt_bboxes, gt_depths)):
            if box.size(0) > 0:
                img_inds = box.new_full((box.size(0), 1), img_id)
                ct = torch.cat([img_inds, box, d[:, None]], dim=-1)
            else:
                ct = box.new_zeros((0, 6))
            centers.append(ct)
        centers = torch.cat(centers, 0)
        
        # calculate IoU for matching
        ious = self.box_iou(rois[:, 1:5], centers[:, 1:5])
        if ious.numel() > 0:
            in_same_img = rois[:, 0:1] == centers[None, :, 0]
            ious[~in_same_img] = 0
            new_ious = ious.clone()
            iou_thr = 0.6
            new_ious[ious < iou_thr] = 0

            # 1 pred can only be matched with up to 1 gt
            max_ious_for_preds = new_ious.max(dim=1, keepdim=True).values
            new_ious[new_ious < max_ious_for_preds] = 0
            # 1 gt can only be matched with up to 1 pred
            max_ious_for_gts = new_ious.max(dim=0, keepdim=True).values
            new_ious[new_ious < max_ious_for_gts] = 0
            inds = (new_ious >= iou_thr).nonzero()
            pred_inds, gt_inds = inds[..., 0], inds[..., 1]

            # get matched prediction and GT
            depth_logits_preds = depth_logits[pred_inds]
            depth_gts = centers[:, 5][gt_inds]
            
            # calculate predicted depth from logits (through probability distribution expectation)
            # clip logits to prevent softmax from producing extreme probability values
            depth_logits_preds = torch.clamp(depth_logits_preds, min=-10, max=10)
            depth_probs = F.softmax(depth_logits_preds, dim=-1)
            pred_depths = (depth_probs @ depth_bin[:, None])[:, 0]
            
            # filter out invalid depth values (avoid log(0))
            # use stricter threshold and ensure predicted depth is within reasonable range
            valid_mask = (pred_depths > 1e-2) & (depth_gts > 1e-2) & (pred_depths < 200) & (depth_gts < 200)
            
            if valid_mask.sum() > 0:
                pred_depths_valid = pred_depths[valid_mask]
                depth_gts_valid = depth_gts[valid_mask]
                
                # clip depth values to ensure within reasonable range
                pred_depths_valid = torch.clamp(pred_depths_valid, min=0.1, max=150)
                depth_gts_valid = torch.clamp(depth_gts_valid, min=0.1, max=150)
                
                # calculate SILog Loss
                # g = log(pred) - log(gt)
                g = torch.log(pred_depths_valid) - torch.log(depth_gts_valid)
                
                g = torch.clamp(g, min=-3, max=3)
                
                Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
                
                loss = 10 * torch.sqrt(Dg + 1e-8)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    return None, None, None
                
                return loss, pred_inds, pred_depths
            else:
                # no valid matches
                return None, None, None
        else:
            # no matches
            return None, None, None

    @staticmethod
    def box_iou(rois_a, rois_b, eps=1e-4):
        rois_a = rois_a[..., None, :]  # [*, n, 1, 4]
        rois_b = rois_b[..., None, :, :]  # [*, 1, m, 4]
        xy_start = torch.maximum(rois_a[..., 0:2], rois_b[..., 0:2])
        xy_end = torch.minimum(rois_a[..., 2:4], rois_b[..., 2:4])
        wh = torch.maximum(xy_end - xy_start, rois_a.new_tensor(0))  # [*, n, m, 2]
        intersect = wh.prod(-1)  # [*, n, m]
        wh_a = rois_a[..., 2:4] - rois_a[..., 0:2]  # [*, m, 1, 2]
        wh_b = rois_b[..., 2:4] - rois_b[..., 0:2]  # [*, 1, n, 2]
        area_a = wh_a.prod(-1)
        area_b = wh_b.prod(-1)
        union = area_a + area_b - intersect
        iou = intersect / (union + eps)
        return iou

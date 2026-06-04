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
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import warnings # Import warnings

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS, build_detector, build_roi_extractor, build_neck, build_head
from mmdet.core import bbox2roi
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.builder import build_query_generator
from mmdet3d.core import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.models.utils.depth_encoder import DepthEncoderResNet

def bbox3d2result(bboxes, scores, labels, logits=None, obj_idxes=None, track_scores=None, attrs=None, pred_history=None, modal=None, query_id=None, modal_weights_img=None, modal_weights_pts=None):
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if logits is not None:
        result_dict['logits'] = logits.cpu()

    if obj_idxes is not None:
        result_dict['track_ids'] = obj_idxes.cpu()
        result_dict['track_scores'] = track_scores.cpu()

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    if pred_history is not None:
        # Move tensors in pred_history to CPU and convert them to lists for JSON serialization.
        result_dict['pred_history'] = {
            'cls_scores': pred_history['cls_scores'].cpu(),  # [num_layers, num_queries, num_classes]
            'bbox_preds': pred_history['bbox_preds'].cpu()   # [num_layers, num_queries, bbox_dim]
        }

    if modal is not None:
        result_dict['modal'] = modal  # Already a list of strings; no conversion needed.
    
    if query_id is not None:
        result_dict['query_id'] = query_id  # Already a list of integers; no conversion needed.
    
    if modal_weights_img is not None:
        result_dict['modal_weights_img'] = modal_weights_img.cpu()
    
    if modal_weights_pts is not None:
        result_dict['modal_weights_pts'] = modal_weights_pts.cpu()

    return result_dict


class GroupedItems(object):
    def __init__(self, items):
        self.items = items

    def __getitem__(self, key):
        return [x[key] for x in self.items]

    def __len__(self):
        return len(self.items)


@DETECTORS.register_module()
class MV2DFusionV2Decouple(MVXTwoStageDetector):
    """MV2D."""

    def __init__(self,
                 dataset='nuscenes',
                 # lidar branch
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 pts_query_generator=None,
                 # image branch
                 img_backbone=None,
                 img_neck=None,
                 img_rpn_head=None,
                 img_roi_head=None,
                 img_roi_head_detach=False,
                 img_roi_extractor=None,
                 img_query_generator=None,
                 use_grid_mask=False,
                 use_2d_proposal=False,
                 # lidar assist image
                 geometric_transfer=False,
                 depth_input_channel=2,
                 img_level_num=5,
                 in_channels_img=256,
                 hidden_channel=256,
                 # fusion head
                 fusion_bbox_head=None,
                 # training
                 gt_mono_loss=False,
                 use_decoupled_loss=False,
                 loss_weight_3d=1.,
                 loss_weight_pts=1.,
                 # config
                 position_level=0,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 # eval settings
                 eval_pts_query=False,
                 eval_pts=False,
                 eval_img=False,
                 eval_decoupled=False, # Whether to use decoupled query processing during inference.
                 trainable_components=None, # Add trainable_components parameter
                 output_features=False, # Whether to output encoder features.
                 use_only_learnable_query=False, # Whether to use only learnable queries without 2D or 3D queries.
                 ):
        self.dataset = dataset
        if pts_voxel_layer is not None:
            self.voxelize_reduce = pts_voxel_layer.pop('voxelize_reduce')

        super(MV2DFusionV2Decouple, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                            pts_middle_encoder, pts_fusion_layer,
                                            img_backbone, pts_backbone, img_neck, pts_neck,
                                            pts_bbox_head, None, img_rpn_head,
                                            train_cfg, test_cfg, pretrained)

        if fusion_bbox_head is not None:
            fusion_train_cfg = train_cfg.fusion if train_cfg else None
            fusion_bbox_head.update(train_cfg=fusion_train_cfg)
            fusion_test_cfg = test_cfg.fusion if test_cfg else None
            fusion_bbox_head.update(test_cfg=fusion_test_cfg)
            self.fusion_bbox_head = build_head(fusion_bbox_head)

        if img_roi_head is not None:
            self.img_roi_head = build_detector(img_roi_head)
            self.img_roi_head_detach = img_roi_head_detach
        if img_roi_extractor is not None:
            self.img_roi_extractor = build_roi_extractor(img_roi_extractor)
        if img_query_generator is not None:
            img_query_generator.update(dict(loss_cls=self.fusion_bbox_head.loss_cls))
            self.img_query_generator = build_query_generator(img_query_generator)

        if pts_query_generator is not None:
            pts_query_generator.update(dict(
                dataset=dataset,
                virtual_voxel_size=self.pts_backbone.virtual_voxel_size,
                point_cloud_range=self.pts_backbone.point_cloud_range,
                head_pc_range=self.fusion_bbox_head.pc_range.tolist(),
            ))
            self.pts_query_generator = build_query_generator(pts_query_generator)

        self.use_2d_proposal = use_2d_proposal
        self.geometric_transfer = geometric_transfer
        self.img_level_num = img_level_num
        if geometric_transfer:
            blocks = [1] * self.img_level_num
            assert len(blocks) == self.img_level_num
            self.depth_resnet = DepthEncoderResNet(depth_input_channel, in_channels_img, hidden_channel, depth_layers=blocks)
        self.gt_mono_loss = gt_mono_loss
        self.use_decoupled_loss = use_decoupled_loss
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.position_level = position_level

        self.loss_weight_3d = loss_weight_3d
        self.loss_weight_pts = loss_weight_pts


        self.eval_pts_query = eval_pts_query
        self.eval_pts = eval_pts
        self.eval_img = eval_img
        self.eval_decoupled = eval_decoupled

        self.trainable_components = trainable_components # Store the parameter
        self._apply_trainable_policy() # Apply the policy at the end of __init__

        self.output_features = output_features # Whether to output encoder features.
        self.use_only_learnable_query = use_only_learnable_query # Whether to use only learnable queries.

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_img_feat(self, img, len_queue=1, training_mode=False, sparse_depth=None):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img = img.squeeze(0)
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        if sparse_depth is not None:
            num_views = 6
            sparse_depth = sparse_depth[:, 0, :, :2]
            sparse_depth = sparse_depth.view(B*num_views, 1, -1, sparse_depth.shape[-2], sparse_depth.shape[-1])
            img_feats = self.depth_resnet(sparse_depth[:, 0], img_feats)
        
        img_feats_det = img_feats

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN / B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN / B / len_queue), C, H, W)
        return img_feats_reshaped, img_feats_det

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=[], out_fp32=True)
    def extract_pts_feat(self, points):
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
            x = self.pts_middle_encoder(feats, coords, batch_size)
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        return x

    def forward_single_frame_train(self,
                                   gt_bboxes_3d=None,
                                   gt_labels_3d=None,
                                   gt_bboxes=None,
                                   gt_labels=None,
                                   img_metas=None,
                                   centers2d=None,
                                   depths=None,
                                   gt_bboxes_ignore=None,
                                   **data):
        data_t = dict()
        for key in data:
            if key in ['instance_inds_2d', 'points', 'pts_feats', 'points_aug']:
                data_t[key] = data[key][0]
            elif key in ['proposals']:
                data_t[key] = data[key][0]
            else:
                data_t[key] = data[key][:, 0]

        losses, outputs = self.forward_pts_train(
            gt_bboxes_3d[0], gt_labels_3d[0], gt_bboxes[0], gt_labels[0],
            img_metas[0], centers2d[0], depths[0],
            requires_grad=True, return_losses=True, **data_t)

        if losses is None:
            return dict(), outputs
        return losses, outputs

    def prepare_detection_data(self, img_metas, **data):
        imgs = data['img']
        if imgs.dim() == 5:
            B, V, C, H, W = imgs.shape
        else:
            B, V, C, H, W = 1, *imgs.shape
        imgs = imgs.flatten(0, 1)
        feats = [x.flatten(0, 1) for x in data['img_feats_for_det']]
        img_metas_det = [dict() for _ in range(B * V)]

        for b in range(B):
            for v in range(V):
                img_meta = {
                    'img_shape': img_metas[b]['img_shape'][v],
                    'ori_shape': img_metas[b]['ori_shape'][:3],
                    'pad_shape': img_metas[b]['pad_shape'][v],
                    'batch_input_shape': (H, W),
                    'scale_factor': img_metas[b]['scale_factor'],
                    'intrinsics': data['intrinsics'][b][v],
                    'extrinsics': data['extrinsics'][b][v],
                    'lidar2img': data['lidar2img'][b][v],
                    'num_views': V,
                }

                if self.training:
                    if 'instance_inds_2d' in data:
                        instance_inds_2d = data['instance_inds_2d'][b][v].clone()
                        img_meta['instance_inds'] = instance_inds_2d

                    img_meta['gt_bboxes'] = data['gt_bboxes'][b][v].clone()
                    img_meta['gt_labels'] = data['gt_labels'][b][v].clone()

                img_metas_det[b * V + v] = img_meta
        return imgs, feats, img_metas_det, (B, V, C, H, W)

    def convert_to_fsd_anno(self, boxes, labels, inv=False):
        if len(labels) == 0:
            return boxes, labels.long()

        b_tensor = boxes.tensor.clone()
        if b_tensor.size(1) == 9:
            vel = b_tensor[:, -2:]
            b_tensor = b_tensor[:, :-2]
        else:
            vel = None

        if self.dataset == 'nuscenes':
            tgt_cls = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                       'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
            fsd_cls = ['car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
                       'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
        else:
            raise NotImplementedError

        tgt_cls = [x.lower() for x in tgt_cls]
        fsd_cls = [x.lower() for x in fsd_cls]

        to_fsd_cls_map = [fsd_cls.index(x) for x in tgt_cls]
        to_tgt_cls_map = [tgt_cls.index(x) for x in fsd_cls]
        if not inv:
            b_tensor[:, 6] = -b_tensor[:, 6] - np.pi / 2
            cls_map = labels.new_tensor(to_fsd_cls_map)
            b_tensor = b_tensor[:, [0, 1, 2, 4, 3, 5, 6]]
        else:
            b_tensor[:, 6] = -(b_tensor[:, 6] + np.pi / 2)
            cls_map = labels.new_tensor(to_tgt_cls_map)
            b_tensor = b_tensor[:, [0, 1, 2, 4, 3, 5, 6]]
        if vel is not None:
            b_tensor = torch.cat([b_tensor, vel], dim=1)
        boxes = boxes.__class__(b_tensor, box_dim=b_tensor.size(-1))
        labels = cls_map[labels]
        return boxes, labels

    @auto_fp16(apply_to=('imgs', 'feats'))
    def forward_roi_head(self, imgs, feats, img_metas):
        dets2d = self.img_roi_head.simple_test_w_feat(feats, img_metas)
        dets = self.process_2d_detections(dets2d, imgs.device)
        return dets2d, dets

    @auto_fp16(apply_to=('imgs', 'feats'))
    def forward_roi_head_train(self, imgs, feats, img_metas, gt_bboxes, gt_labels):
        gt_bboxes = sum(gt_bboxes, [])
        gt_labels = sum(gt_labels, [])
        valid_inds = imgs.new_zeros(len(gt_bboxes), dtype=torch.bool)
        gt_bboxes_valid, gt_labels_valid, img_metas_valid = [], [], []
        for i in range(len(gt_bboxes)):
            if len(gt_bboxes[i]) > 0:
                gt_bboxes_valid.append(gt_bboxes[i])
                gt_labels_valid.append(gt_labels[i])
                img_metas_valid.append(img_metas[i])
                valid_inds[i] = 1

        if not valid_inds.any():
            # grad for all parameters
            gt_bboxes_valid = [imgs.new_tensor([[40, 120, 40, 120]])]
            gt_labels_valid = [imgs.new_tensor([0], dtype=torch.int64)]
            losses = self.img_roi_head.forward_train_w_feat(
                [x[:1] for x in feats], imgs[:1], img_metas[:1], gt_bboxes_valid, gt_labels_valid, )
            losses = {k: ([x * 0 for x in v] if isinstance(v, (list, tuple)) else v * 0) for k, v in losses.items()}

            for k, v in losses.items():
                if isinstance(v, torch.Tensor) and v.isnan().any():
                    losses[k] = v.nan_to_num()
        else:
            losses = self.img_roi_head.forward_train_w_feat(
                [x[valid_inds] for x in feats], imgs[valid_inds], img_metas_valid, gt_bboxes_valid, gt_labels_valid, )
        return losses

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

    def process_2d_gt(self, gt_bboxes, gt_labels, device):
        return [torch.cat(
            [bboxes.to(device), torch.ones([len(labels), 1], dtype=bboxes.dtype, device=device),
             labels.unsqueeze(-1).to(bboxes.dtype)], dim=-1).to(device)
                for bboxes, labels in zip(gt_bboxes, gt_labels)]

    def complement_2d_gt(self, detections, gts, thr=0.6, with_id=False):
        # detections: [n, 6], gts: [m, 6]
        if len(detections) == 0:
            if len(gts) == 0:
                gts = gts.new_zeros([0, 6])
            if with_id:
                gts = torch.cat([gts, torch.zeros_like(gts[..., :1]) - 1], dim=-1)
            return gts
        if len(gts) == 0:
            return detections
        iou = self.box_iou(gts, detections)
        max_iou = iou.max(-1)[0]
        complement_ids = max_iou <= thr
        min_bbox_size = self.img_roi_head.test_cfg.get('min_bbox_size', 0)
        wh = gts[:, 2:4] - gts[:, 0:2]
        valid_ids = (wh >= min_bbox_size).all(dim=1)
        complement_gts = gts[complement_ids & valid_ids]
        if with_id:
            complement_gts = torch.cat([complement_gts, torch.zeros_like(complement_gts[..., :1]) - 1], dim=-1)
        return torch.cat([detections, complement_gts], dim=0)

    def process_2d_detections(self, results, device):
        """
        :param results:
            results: list[per_cls_res] of size BATCH_SIZE
            per_cls_res: list(boxes) of size NUM_CLASSES
            boxes: ndarray of shape [num_boxes, 5->(x1, y1, x2, y2, score)]
        :return:
            detections: list[ndarray of shape [num_boxes, 6->(x1, y1, x2, y2, score, label_id)]] of size len(results)
        """
        detections = [torch.cat(
            [torch.cat([
                torch.tensor(boxes, device=device),
                torch.full((len(boxes), 1), label_id, dtype=torch.float, device=device)], dim=1)
                # if len(boxes) > 0 else torch.zeros((0, 6), device=device)
                for label_id, boxes in enumerate(res)], dim=0) for res in results]
        min_bbox_size = self.img_roi_head.test_cfg.get('min_bbox_size', 0)
        if min_bbox_size > 0:
            new_detections = []
            for det in detections:
                wh = det[:, 2:4] - det[:, 0:2]
                valid = (wh >= min_bbox_size).all(dim=1)
                new_detections.append(det[valid])
            detections = new_detections
        return detections

    def extract_roi_feats(self, feats_det, rois, **data):
        roi_feats = self.img_roi_extractor(feats_det[:self.img_roi_extractor.num_inputs], rois)
        return roi_feats

    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          # Parameters for query reuse
                          reuse_queries=None,  # Dict with 'dyn_query', 'dyn_feats', 'pts_query_center', 'pts_query_feat', 'pts_feat', 'pts_pos'
                          **data):
        data['gt_bboxes'] = gt_bboxes
        data['gt_labels'] = gt_labels
        data['gt_bboxes_3d'] = gt_bboxes_3d
        data['gt_labels_3d'] = gt_labels_3d
        data['depths'] = depths

        imgs_det, feats_det, img_metas_det, imgs_shape = self.prepare_detection_data(img_metas, **data)

        B, V = imgs_shape[:2]
        if not requires_grad:
            raise NotImplementedError
        else:
            # image query generation
            if self.with_img_roi_head:
                losses_det2d = self.forward_roi_head_train(imgs_det, feats_det, img_metas_det, gt_bboxes, gt_labels)

            # Check if we should reuse queries from another branch
            if reuse_queries is not None:
                # Reuse pre-computed queries
                dyn_query = reuse_queries.get('dyn_query')
                dyn_feats = reuse_queries.get('dyn_feats')
                dyn_feats_pred = reuse_queries.get('dyn_feats_pred', dyn_feats)
                pts_query_feat = reuse_queries.get('pts_query_feat')
                pts_query_center = reuse_queries.get('pts_query_center')
                pts_feat = reuse_queries.get('pts_feat')
                pts_pos = reuse_queries.get('pts_pos')
                losses_pts = reuse_queries.get('losses_pts', {})
                n_rois_per_batch = reuse_queries.get('n_rois_per_batch', [0] * B)
            else:
                # Generate queries as normal
                self.eval()
                with torch.no_grad():
                    dets2d, dets = self.forward_roi_head(imgs_det, feats_det, img_metas_det)
                    dets_gt = self.process_2d_gt(sum(gt_bboxes, []), sum(gt_labels, []), imgs_det.device)
                    assert len(dets) == (len(dets_gt))
                    if self.use_2d_proposal:
                        dets = sum([x for x in data['proposals']], [])
                    dets = [self.complement_2d_gt(det, det_gt, )
                            for det, det_gt in zip(dets, dets_gt)]

                    # prevent empty detection during training
                    if sum([len(p) for p in dets]) == 0:
                        proposal = torch.tensor([[10, 20, 30, 40, 0, 1]], dtype=dets[0].dtype, device=dets[0].device)
                        dets = [proposal] + dets[1:]

                    rois = bbox2roi(dets)
                self.train()

                roi_feats = self.extract_roi_feats(feats_det, rois, **data)
                if self.img_roi_head_detach:
                    roi_feats = roi_feats.detach()
                n_rois_per_view = [len(p) for p in dets]
                n_rois_per_batch = [sum(n_rois_per_view[i * V: (i + 1) * V]) for i in range(B)]

                if self.use_only_learnable_query:
                    dyn_query, dyn_feats = None, None
                    dyn_feats_pred = None
                else:
                    dyn_query, dyn_feats = self.img_query_generator(roi_feats, dets, img_metas_det,
                                                                    n_rois_per_view=n_rois_per_view,
                                                                    n_rois_per_batch=n_rois_per_batch,
                                                                    data=data)
                    dyn_feats_pred = dyn_feats

                if self.gt_mono_loss and not self.use_only_learnable_query:
                    rois_gt = bbox2roi(dets_gt)
                    roi_feats_gt = self.extract_roi_feats(feats_det, rois_gt, **data)
                    n_rois_per_view_gt = [len(p) for p in dets_gt]
                    n_rois_per_batch_gt = [sum(n_rois_per_view_gt[i * V: (i + 1) * V]) for i in range(B)]
                    _, dyn_feats = self.img_query_generator(roi_feats_gt, dets_gt, img_metas_det,
                                                            n_rois_per_view=n_rois_per_view_gt,
                                                            n_rois_per_batch=n_rois_per_batch_gt,
                                                            data=data,)
                    n_rois_per_batch = n_rois_per_batch_gt

                # lidar query generation            
                # Convert gt_bboxes_3d to ISFusion format: [x,y,z,l,w,h,yaw] -> [x,y,z,w,l,h,-yaw-pi/2]
                gt_bboxes_3d_isfusion = []
                for bboxes in gt_bboxes_3d:
                    # if len(bboxes) > 0:
                    boxes_tensor = bboxes.tensor.clone()
                    # Swap l,w (dims 3,4)
                    boxes_tensor[:, [3, 4]] = boxes_tensor[:, [4, 3]]
                    # Convert yaw: yaw -> -yaw - pi/2
                    boxes_tensor[:, 6] = -boxes_tensor[:, 6] - np.pi / 2
                    gt_bboxes_3d_isfusion.append(bboxes.__class__(boxes_tensor, box_dim=boxes_tensor.size(-1)))
                    # else:
                    #     gt_bboxes_3d_isfusion.append(bboxes)
                
                out_dict = self.pts_backbone.forward_train(data['pts_feats'], img_metas, gt_bboxes_3d_isfusion, gt_labels_3d, img=data['img'], lidar2img=data['lidar2img'])
                
                # Always generate point cloud features for cross-attention, while optionally not using point queries.
                pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
                    out_dict['lidar_feats'], out_dict['query_feats'], out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], B)
                losses_pts = out_dict['losses']
                
                # If only learnable queries are used, skip the generated point queries.
                if self.use_only_learnable_query:
                    pts_query_feat, pts_query_center = None, None

        if return_losses:
            losses = {}
            # lidar prior fusion info
            if dyn_feats is not None and 'fusion_weight' in dyn_feats:
                losses['lidar_prior_img_weight'] = torch.ones(1).cuda() * (dyn_feats['fusion_weight'])
                losses['lidar_prior_pts_weight'] = torch.ones(1).cuda() * (1 - dyn_feats['fusion_weight'])
            # fusion head loss
            outs_pts, outs_img = None, None
            if self.use_decoupled_loss:
                # 1. Point cloud query only
                outs_pts = self.fusion_bbox_head(img_metas, dyn_query=None, dyn_feats=None,
                                                  pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                                  pts_feat=pts_feat, pts_pos=pts_pos, mode='pts', **data)
                loss_inputs_pts = [gt_bboxes_3d, gt_labels_3d, outs_pts, None, img_metas]
                losses_pts_head = self.fusion_bbox_head.loss(*loss_inputs_pts, branch='pts')
                for k, v in losses_pts_head.items():
                    losses['query3d.' + k] = v

                # 2. Image query only
                outs_img = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats_pred,
                                                  pts_query_center=None, pts_query_feat=None,
                                                  pts_feat=pts_feat, pts_pos=pts_pos, mode='img', **data)
                loss_inputs_img = [gt_bboxes_3d, gt_labels_3d, outs_img, None, img_metas]
                losses_img_head = self.fusion_bbox_head.loss(*loss_inputs_img, branch='img')
                for k, v in losses_img_head.items():
                    losses['query2d.' + k] = v
            
            # 3. All queries (calculated always)
            outs_all = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats_pred,
                                          pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                          pts_feat=pts_feat, pts_pos=pts_pos, mode='all', **data)
            loss_inputs_all = [gt_bboxes_3d, gt_labels_3d, outs_all, None, img_metas]
            main_losses = self.fusion_bbox_head.loss(*loss_inputs_all, log_stats=True, branch='all')
            losses.update(main_losses)

            # point cloud detector loss
            for k, v in losses_pts.items():
                if 'loss' in k:
                    v = v * self.loss_weight_pts
                losses['pts.' + k] = v

            # image detector loss
            if self.with_img_roi_head:
                for k, v in losses_det2d.items():
                    losses['det2d.' + k] = v

            # image query generator auxiliary loss
            if dyn_feats is not None:
                losses_img_qg = dict()

                # monodepth loss
                if 'd_loss' in dyn_feats:
                    losses_img_qg['d_loss'] = dyn_feats['d_loss']
                elif 'd_silog_loss' in dyn_feats:
                    losses_img_qg['d_silog_loss'] = dyn_feats['d_silog_loss']

                # query generator loss
                if dyn_feats.get('cls_scores', None) is not None:
                    cls_scores = dyn_feats['cls_scores'].split(n_rois_per_batch, dim=0)
                    bbox_preds = dyn_feats['bbox_preds'].split(n_rois_per_batch, dim=0)
                    gt_bboxes_3d_ = [torch.cat((b.gravity_center, b.tensor[:, 3:]),
                                               dim=1).to(imgs_det.device).clone() for b in gt_bboxes_3d]
                    for x in gt_bboxes_3d_:
                        x[:, 6:] = 0
                    loss_cls, loss_bbox, *_ = self.fusion_bbox_head.loss_single(cls_scores, bbox_preds, gt_bboxes_3d_,
                                                                         gt_labels_3d, branch='img')
                    losses_img_qg.update({'loss_cls': loss_cls, 'loss_bbox': loss_bbox})

                for k, v in losses_img_qg.items():
                    losses['imgqg.' + k] = v

            # loss scaling
            for k, v in losses.items():
                if 'loss' in k:
                    if 'det2d.' not in k:
                        losses[k] = v * self.loss_weight_3d

            # Store query information in outputs for potential reuse
            outs_all['dyn_query'] = dyn_query
            outs_all['dyn_feats'] = dyn_feats
            outs_all['dyn_feats_pred'] = dyn_feats_pred
            outs_all['pts_query_feat'] = pts_query_feat
            outs_all['pts_query_center'] = pts_query_center
            outs_all['pts_feat'] = pts_feat
            outs_all['pts_pos'] = pts_pos
            outs_all['losses_pts'] = losses_pts
            outs_all['n_rois_per_batch'] = n_rois_per_batch

            return losses, outs_all
        else:
            return None, None

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        B, T, V, _, H, W = data['img'].shape
        if T != 1:
            raise AssertionError(f'MV2DFusionV2Decouple release config expects single-frame input, got T={T}')

        if self.geometric_transfer:
            img_feats, img_feats_for_det = self.extract_img_feat(data['img'], 1, sparse_depth=data['sparse_depth'])
        else:
            img_feats, img_feats_for_det = self.extract_img_feat(data['img'], 1)

        data['img_feats'] = img_feats
        data['img_feats_for_det'] = GroupedItems([x.view(B, T, V, *x.shape[1:]) for x in img_feats_for_det])
        data['pts_feats'] = data['points']

        losses, output = self.forward_single_frame_train(
            gt_bboxes_3d, gt_labels_3d, gt_bboxes, gt_labels, img_metas,
            centers2d, depths, gt_bboxes_ignore, **data)

        return losses, output

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""

        imgs_det, feats_det, img_metas_det, imgs_shape = self.prepare_detection_data(img_metas, **data)
        B, V = imgs_shape[:2]

        # image query generation
        dets2d, dets = self.forward_roi_head(imgs_det, feats_det, img_metas_det)
        if self.use_2d_proposal:
            dets = data['proposals']

        if sum([len(p) for p in dets]) == 0:
            proposal = torch.tensor([[0, 50, 50, 100, 0, 1]], dtype=dets[0].dtype,
                                    device=dets[0].device)
            dets = [proposal] + dets[1:]
        rois = bbox2roi(dets)

        roi_feats = self.extract_roi_feats(feats_det, rois, **data)
        n_rois_per_view = [len(p) for p in dets]
        n_rois_per_batch = [sum(n_rois_per_view[i * V: (i + 1) * V]) for i in range(B)]
        if 'sparse_depth' in data:
            data['sparse_depth'] = data['sparse_depth'][0]

        if self.use_only_learnable_query:
            dyn_query, dyn_feats = None, None
        else:
            dyn_query, dyn_feats = self.img_query_generator(roi_feats, dets, img_metas_det,
                                                        n_rois_per_view=n_rois_per_view,
                                                        n_rois_per_batch=n_rois_per_batch,
                                                        data=data)

        # lidar query generation
        out_dict = self.pts_backbone.simple_test(data['pts_feats'], img_metas, img=data['img'], lidar2img=data['lidar2img'])
        if self.eval_pts_query:
            bbox_list = out_dict['query_bboxes']
            bbox_results = [
                bbox3d2result(*bbox)
                for bbox in bbox_list
            ]
            return bbox_results
        
        # Always generate point cloud features for cross-attention, while optionally not using point queries.
        pts_feat, pts_pos, pts_query_feat, pts_query_center = self.pts_query_generator(
            out_dict['lidar_feats'], out_dict['query_feats'], out_dict['query_xyz'], out_dict['query_pred'], out_dict['query_cat'], B)
        
        # If only learnable queries are used, skip the generated point queries.
        if self.use_only_learnable_query:
            pts_query_feat, pts_query_center = None, None

        # Generate outputs for all three branches during inference as well, matching training.
        outs_pts, outs_img = None, None
        
        # Use eval_decoupled to decide whether to generate all three branches.
        if self.eval_decoupled:
            # 1. Point cloud query only
            outs_pts = self.fusion_bbox_head(img_metas, dyn_query=None, dyn_feats=None,
                                              pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                              pts_feat=pts_feat, pts_pos=pts_pos, mode='pts', eval_decoupled=self.eval_decoupled, **data)

            # 2. Image query only
            outs_img = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats,
                                              pts_query_center=None, pts_query_feat=None,
                                              pts_feat=pts_feat, pts_pos=pts_pos, mode='img', eval_decoupled=self.eval_decoupled, **data)

        # Determine the final mode to use.
        if self.eval_img:
            mode = 'img'
        elif self.eval_pts:
            mode = 'pts'
        else:
            mode = 'all'
            
        # 3. All queries (always generated)
        outs = self.fusion_bbox_head(img_metas, dyn_query=dyn_query, dyn_feats=dyn_feats,
                                     pts_query_center=pts_query_center, pts_query_feat=pts_query_feat,
                                     pts_feat=pts_feat, pts_pos=pts_pos, mode=mode, eval_decoupled=self.eval_decoupled,
                                     **data)
                
        bbox_list = self.fusion_bbox_head.get_bboxes(
            outs, img_metas)
        
        bbox_results = []
        for result in bbox_list:
            bbox_results.append(
                bbox3d2result(
                    bboxes=result['boxes_3d'],
                    scores=result['scores_3d'],
                    labels=result['labels_3d'],
                    logits=result.get('logits'),
                    pred_history=result.get('pred_history'),
                    modal=result.get('modal'),
                    query_id=result.get('query_id'),
                    modal_weights_img=result.get('modal_weights_img'),
                    modal_weights_pts=result.get('modal_weights_pts'),
                )
            )
        return bbox_results

    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        B, V, _, H, W = data['img'].shape
        if self.geometric_transfer:
            img_feats_reshaped, img_feats_for_det = self.extract_img_feat(data['img'], 1, sparse_depth=data['sparse_depth'])
        else:
            img_feats_reshaped, img_feats_for_det = self.extract_img_feat(data['img'], 1)

        data['img_feats'] = img_feats_reshaped
        data['img_feats_for_det'] = [x.view(B, V, *x.shape[1:]) for x in img_feats_for_det]

        rec_points = [data['points'].squeeze(0)]
        data['pts_feats'] = rec_points

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list

    def forward_test(self, img_metas, rescale, **data):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(name, type(var)))

        for key in data:
            if key in ['instance_inds_2d']:
                data[key] = data[key][0][0]
            elif key in ['proposals']:
                data[key] = data[key][0][0]
            elif key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:  # key == 'img'
                data[key] = data[key][0]

        for i in range(len(img_metas[0])):
            img_metas[0][i]['lidar2img'] = data['lidar2img'][i].cpu().numpy()
            img_metas[0][i]['intrinsics'] = data['intrinsics'][i].cpu().numpy()
            img_metas[0][i]['extrinsics'] = data['extrinsics'][i].cpu().numpy()
        results = self.simple_test(img_metas[0], **data)
        return results

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        for key in ['proposals', 'instance_inds_2d', 'points', 'points_aug']:
            if key in data:
                data[key] = list(zip(*data[key]))

        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                if key in data:
                    data[key] = list(zip(*data[key]))
            return self.forward_train(**data)[0]
        else:
            return self.forward_test(**data)

    def _apply_trainable_policy(self):
        if self.trainable_components is None:
            # If None, all parameters are trainable by default (standard behavior)
            return

        # First, freeze all parameters in the model
        for param in self.parameters():
            param.requires_grad = False
        
        warnings.warn(
            f"Applying trainable policy. Initially freezing all parameters. "
            f"Will attempt to unfreeze components specified by paths in: {self.trainable_components}"
        )

        unfrozen_count = 0  # Counts paths that led to unfreezing params
        
        # Convert to set for faster lookups
        trainable_paths_set = set(self.trainable_components)
        all_module_names_in_model = set()

        for name, module in self.named_modules():
            all_module_names_in_model.add(name)
            if name in trainable_paths_set:
                # Get all parameters recursively within this module
                params_in_module = list(module.parameters()) 

                if not params_in_module:
                    warnings.warn(
                        f"Component path '{name}' resolved to an nn.Module, but it (and its submodules) "
                        f"contains no parameters to unfreeze."
                    )
                else:
                    for param in params_in_module:
                        param.requires_grad = True
                    warnings.warn(f"Successfully unfroze parameters for component path: '{name}'")
                    unfrozen_count += 1
        
        # Check for paths in trainable_components that were not found
        for path_str in self.trainable_components:
            if path_str not in all_module_names_in_model:
                 warnings.warn(
                     f"Specified trainable component path '{path_str}' was not found in the model's named modules and was not processed."
                 )

        if unfrozen_count == 0 and self.trainable_components: # trainable_components is not None and not empty
            warnings.warn(
                f"Trainable policy was applied with paths {self.trainable_components}, "
                f"but no parameters were successfully unfrozen (either paths not found or modules had no parameters). "
                f"All model parameters remain frozen."
            )
        elif unfrozen_count > 0:
            warnings.warn(
                f"Successfully unfroze parameters for {unfrozen_count} component path(s). "
                f"Other parameters remain frozen."
            )
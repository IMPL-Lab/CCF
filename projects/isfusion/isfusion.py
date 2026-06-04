import mmcv
import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.core.bbox import limit_period

@DETECTORS.register_module()
class ISFusionDetector(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 norm_eval=False,
                 freeze=False,
                 pc_range=None,
                 voxel_size=None,
                 out_size_factor=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        super(ISFusionDetector, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                                               pts_middle_encoder, pts_fusion_layer,
                                               img_backbone, pts_backbone, img_neck, pts_neck,
                                               pts_bbox_head, img_roi_head, img_rpn_head,
                                               train_cfg, test_cfg, pretrained, init_cfg, **kwargs)

        self.detach = kwargs.get('detach', False)

        out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.virtual_voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_cloud_range = pc_range
        self.pillar_size = [voxel_size[0]*out_size_factor, voxel_size[1]*out_size_factor, self.pc_range[5]-self.pc_range[2]]

        self.pts_pillar_layer = Voxelization(
            max_num_points=self.fusion_encoder.num_points_in_pillar,
            voxel_size=self.pillar_size,
            max_voxels=(30000, 60000),
            point_cloud_range=self.pc_range)

        self.norm_eval = norm_eval
        self.freeze = freeze

    def init_weights(self) -> None:
        super(ISFusionDetector, self).init_weights()
        if self.freeze:
            for p in self.parameters():
                p.requires_grad = False

    def train(self, mode=True):
        from torch.nn.modules.batchnorm import _BatchNorm
        super(ISFusionDetector, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if 'img_mask_idx' in img_metas[0].keys():
            for i in range(len(img_metas)):
                this_mask_idx = img_metas[i]['img_mask_idx']
                if not this_mask_idx[0] == -1:
                    img[i][this_mask_idx, ...] = 0.0

        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            assert img.dim() == 5
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)

            img_feats = self.img_backbone(img.float()) # [800, 1440] -> [100, 180]
        else:
            return None

        if self.detach:
            img_feats = [img_feat.detach() for img_feat in img_feats]

        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        return img_feats

    def isfusion(self, pts, pts_feats, img_feats, img_metas, batch_size, **kwargs):

        # create BEV space
        pillars, pillars_num_points, pillar_coors = self.voxelize(pts, voxel_type='pillar')
        pts_metas = {}
        pts_metas['pillars'] = pillars
        pts_metas['pillars_num_points'] = pillars_num_points
        pts_metas['pillar_coors'] = pillar_coors
        pts_metas['pts'] = pts
        pts_metas['pillar_size'] = self.pillar_size

        kwargs.update(dict(pts_metas=pts_metas))
        kwargs.update(dict(img_metas=img_metas))
        kwargs.update(dict(pts_backbone=self.pts_backbone))

        x = self.fusion_encoder(img_feats, pts_feats, batch_size, **kwargs)

        return x

    def extract_pts_feat(self, pts, img_feats, img_metas, **kwargs):
        """Extract features of points."""
        if not self.with_pts_bbox:
            return None

        voxels, coors = self.dynamic_voxelize(pts)
        voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors, pts, img_feats, img_metas)
        batch_size = coors[-1, 0].item() + 1
        x, _, kwargs = self.pts_middle_encoder(voxel_features, feature_coors, batch_size, **kwargs)

        x, ins_heatmap = self.isfusion(pts, x, img_feats, img_metas, batch_size, **kwargs)

        if self.with_pts_neck:
            x = self.pts_neck(x)

        if self.training:
            return x, ins_heatmap
        else:
            return x

    @torch.no_grad()
    @force_fp32()
    def dynamic_voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.pts_voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []

        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, voxel_type='voxel'):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            if voxel_type == 'pillar':
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(res)
            else:
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas, **kwargs)
        return (img_feats, pts_feats)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses and intermediate features.
        """
        # if self.training:
        #     torch.cuda.empty_cache()

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        losses = dict()

        output_dict = dict(losses={})
        
        if pts_feats:
            losses_pts, outs, bbox_pts = self.forward_pts_train(
                pts_feats, img_feats, gt_bboxes_3d, gt_labels_3d, 
                img_metas, gt_bboxes_ignore, return_feats=True)
            
            losses.update(losses_pts)

            device = points[0].device
            query_xyz_list = []
            query_pred_list = []
            query_bboxes_list = []
            
            for i, pts_bbox in enumerate(bbox_pts):
                query_xyz_list.append(pts_bbox['boxes_3d'].tensor[:, :3].to(device))
                query_pred_list.append(torch.cat([pts_bbox['boxes_3d'].tensor[:, 3:], 
                                               pts_bbox['scores_3d'][:, None]], dim=1).to(device))
                query_bboxes_list.append([pts_bbox['boxes_3d'], pts_bbox['scores_3d'], pts_bbox['labels_3d']])

            output_dict.update(dict(
                lidar_feats=outs['lidar_feats'].permute(0, 2, 3, 1),
                query_xyz=query_xyz_list,
                query_cat=outs['query_cat'],
                query_feats=outs['query_feats'].permute(0, 2, 1),
                query_pred=query_pred_list,
                query_bboxes=query_bboxes_list
            ))
        
        if img_feats:
            losses_img = self.forward_img_train(
                img_feats,
                img_metas=img_metas,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposals=proposals)
            losses.update(losses_img)
        
        # 更新损失
        output_dict['losses'] = losses
        
        return output_dict

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          return_feats=False):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            return_feats (bool): Whether to return intermediate features.
                Default to False.

        Returns:
            dict or tuple: When return_feats is False, returns losses only.
                When return_feats is True, returns losses and intermediate features.
        """
        if len(pts_feats) == 2:  # instance heatmap loss
            outs = self.pts_bbox_head(pts_feats[0], img_feats, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, pts_feats[1]]
        else:
            outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        
        losses = self.pts_bbox_head.loss(*loss_inputs)
        
        if return_feats:
            with torch.no_grad():
                bbox_pts = []
                for i in range(len(img_metas)):
                    outs_single = {}
                    for key in outs[0][0].keys():
                        outs_single[key] = outs[0][0][key][i].unsqueeze(0)
                    outs_single = [[outs_single]]
                    bbox_pt = {}
                    # Assuming get_bboxes returns results in the old coordinate system
                    res = self.pts_bbox_head.get_bboxes(
                        outs_single, [img_metas[i]], rescale=False, batch_id=i)[0]
                    
                    boxes = res[0] # Get the BaseInstance3DBoxes object
                    scores = res[1]
                    labels = res[2]

                    # Apply coordinate system adjustments for mmdet3d v1.0.0+
                    if boxes.tensor.shape[0] > 0: # Check if there are any boxes
                        # Swap x_size and y_size (indices 3 and 4)
                        boxes.tensor[:, [3, 4]] = boxes.tensor[:, [4, 3]]
                        # Negate yaw angle (index 6) and normalize to [-pi, pi]
                        yaw = -boxes.tensor[:, 6] - torch.pi / 2
                        # Normalize angle
                        boxes.tensor[:, 6] = limit_period(yaw, period=torch.pi * 2)

                    bbox_pt['boxes_3d'] = boxes
                    bbox_pt['scores_3d'] = scores
                    bbox_pt['labels_3d'] = labels
                    bbox_pts.append(bbox_pt)

            return losses, outs[0][0], bbox_pts
        
        return losses

    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, x_img, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)

        # Apply coordinate system adjustments for mmdet3d v1.0.0+
        for i in range(len(bbox_list)):
            if bbox_list[i][0].tensor.shape[0] > 0: # Check if there are any boxes
                # Swap x_size and y_size (indices 3 and 4)
                bbox_list[i][0].tensor[:, [3, 4]] = bbox_list[i][0].tensor[:, [4, 3]]
                # Negate yaw angle (index 6) and normalize to [-pi, pi]
                yaw = -bbox_list[i][0].tensor[:, 6] - torch.pi / 2
                # Normalize angle
                bbox_list[i][0].tensor[:, 6] = limit_period(yaw, period=torch.pi * 2)

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results, outs[0][0]

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""

        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)

        if pts_feats and self.with_pts_bbox:
            bbox_pts, outs = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            
            # pts_bboxes = []
            # lidar_feats_list = []
            query_xyz_list = []
            # query_cat_list = []
            # query_feats_list = []
            query_pred_list = []
            query_bboxes_list = []
            
            device = points[0].device
            for i, pts_bbox in enumerate(bbox_pts):
                # pts_bboxes.append(pts_bbox)
                # lidar_feats_list.append(out['lidar_feat'].permute(0, 2, 3, 1))
                query_xyz_list.append(pts_bbox['boxes_3d'].tensor[:, :3].to(device))
                # query_cat_list.append(out['query_cat'])
                # query_feats_list.append(out['query_feats'])
                query_pred_list.append(torch.cat([pts_bbox['boxes_3d'].tensor[:, 3:], pts_bbox['scores_3d'][:, None]], dim=1).to(device))
                query_bboxes_list.append([pts_bbox['boxes_3d'], pts_bbox['scores_3d'], pts_bbox['labels_3d']])
            
            output_dict = dict(
                # pts_bbox=pts_bboxes,
                lidar_feats=outs['lidar_feats'].permute(0, 2, 3, 1),
                query_xyz=query_xyz_list,
                query_cat=outs['query_cat'],
                query_feats=outs['query_feats'].permute(0, 2, 1),
                query_pred=query_pred_list,
                query_bboxes=query_bboxes_list
            )
            
            return output_dict
        else:
            return None
# Copyright (c) Wang, Z
# ------------------------------------------------------------------------
# Modified from FAR3D https://github.com/megvii-research/Far3D
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from PETR (https://github.com/megvii-research/PETR)
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

import copy
import numpy as np
import mmcv
from mmdet.datasets.builder import PIPELINES
import torch
from PIL import Image
import os
import random
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox

@PIPELINES.register_module()
class PadMultiViewImage():
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """
    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert size is not None or size_divisor is not None
        assert size_divisor is None or size is None
    
    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [mmcv.impad(img,
                                shape = self.size, pad_val=self.pad_val) for img in results['img']]
        elif self.size_divisor is not None:
            padded_img = [mmcv.impad_to_multiple(img,
                                self.size_divisor, pad_val=self.pad_val) for img in results['img']]
        results['img_shape'] = [img.shape for img in results['img']]
        results['img'] = padded_img
        results['pad_shape'] = [img.shape for img in padded_img]
        results['pad_fix_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
    
    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results


    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results['img'] = [mmcv.imnormalize(
            img, self.mean, self.std, self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class ResizeCropFlipRotImage():
    def __init__(self, data_aug_conf=None, with_2d=True, filter_invisible=True, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training
        self.min_size = 2.0
        self.with_2d = with_2d
        self.filter_invisible = filter_invisible

        self.cached_scene_augs = dict()
        self.scene_next_token = dict()

    def __call__(self, results):

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        new_gt_bboxes = []
        new_centers2d = []
        new_gt_labels = []
        new_depths = []
        new_proposals = []
        new_gt_inds = []

        assert self.data_aug_conf['rot_lim'] == (0.0, 0.0), "Rotation is not currently supported"
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()

        for i in range(N):
            img = Image.fromarray(np.uint8(imgs[i]))
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            if self.with_2d: # sync_2d bbox labels
                if 'gt_bboxes' in results:
                    gt_bboxes = results['gt_bboxes'][i]
                    centers2d = results['centers2d'][i]
                    gt_labels = results['gt_labels'][i]
                    depths = results['depths'][i]
                    gt_inds = None if 'instance_inds_2d' not in results else results['instance_inds_2d'][i]
                    if len(gt_bboxes) != 0:
                        gt_bboxes, centers2d, gt_labels, depths, gt_inds = self._bboxes_transform(
                            gt_bboxes,
                            centers2d,
                            gt_labels,
                            depths,
                            resize=resize,
                            crop=crop,
                            flip=flip,
                            gt_inds=gt_inds,
                        )
                    if len(gt_bboxes) != 0 and self.filter_invisible:
                        gt_bboxes, centers2d, gt_labels, depths, gt_inds = self._filter_invisible(gt_bboxes, centers2d, gt_labels, depths, gt_inds=gt_inds)

                    new_gt_bboxes.append(gt_bboxes)
                    new_centers2d.append(centers2d)
                    new_gt_labels.append(gt_labels)
                    new_depths.append(depths)
                    new_gt_inds.append(gt_inds)

                if 'proposals' in results:
                    proposals = results['proposals'][i]
                    if len(proposals) > 0:
                        proposals = self._proposals_transform(proposals, resize, crop, flip)
                        if self.filter_invisible:
                            fH, fW = self.data_aug_conf["final_dim"]
                            proposal_boxes = proposals[..., :4]
                            xy_max = np.minimum(proposal_boxes[..., 2:4], [fW-1, fH-1])
                            xy_min = np.maximum(proposal_boxes[..., 0:2], [0, 0])
                            wh = np.maximum(xy_max - xy_min, 0)
                            keep = (wh >= self.min_size).all(-1)
                            proposals = proposals[keep]
                    new_proposals.append(proposals)

            new_imgs.append(np.array(img).astype(np.float32))
            results['intrinsics'][i][:3, :3] = ida_mat @ results['intrinsics'][i][:3, :3]
        results['gt_bboxes'] = new_gt_bboxes
        results['centers2d'] = new_centers2d
        results['gt_labels'] = new_gt_labels
        results['depths'] = new_depths
        results['img'] = new_imgs
        results['lidar2img'] = [results['intrinsics'][i] @ results['extrinsics'][i] for i in range(len(results['extrinsics']))]
        if 'proposals' in results:
            results['proposals'] = new_proposals
        if 'instance_inds_2d' in results:
            results['instance_inds_2d'] = new_gt_inds
        return results

    def _proposals_transform(self, proposals, resize, crop, flip):
        bboxes = proposals[:, :4]
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)

        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1

        proposals = np.concatenate([bboxes, proposals[:, 4:]], axis=1)
        proposals = proposals[keep]
        return proposals

    def _bboxes_transform(self, bboxes, centers2d, gt_labels, depths, resize, crop, flip, gt_inds=None):
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        fH, fW = self.data_aug_conf["final_dim"]
        bboxes = bboxes * resize
        bboxes[:, 0] = bboxes[:, 0] - crop[0]
        bboxes[:, 1] = bboxes[:, 1] - crop[1]
        bboxes[:, 2] = bboxes[:, 2] - crop[0]
        bboxes[:, 3] = bboxes[:, 3] - crop[1]
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, fW)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, fW)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, fH) 
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, fH)
        keep = ((bboxes[:, 2] - bboxes[:, 0]) >= self.min_size) & ((bboxes[:, 3] - bboxes[:, 1]) >= self.min_size)


        if flip:
            x0 = bboxes[:, 0].copy()
            x1 = bboxes[:, 2].copy()
            bboxes[:, 2] = fW - x0
            bboxes[:, 0] = fW - x1
        bboxes = bboxes[keep]

        centers2d  = centers2d * resize
        centers2d[:, 0] = centers2d[:, 0] - crop[0]
        centers2d[:, 1] = centers2d[:, 1] - crop[1]
        centers2d[:, 0] = np.clip(centers2d[:, 0], 0, fW)
        centers2d[:, 1] = np.clip(centers2d[:, 1], 0, fH) 
        if flip:
            centers2d[:, 0] = fW - centers2d[:, 0]

        centers2d = centers2d[keep]
        gt_labels = gt_labels[keep]
        depths = depths[keep]
        if gt_inds is not None:
            gt_inds = gt_inds[keep]

        return bboxes, centers2d, gt_labels, depths, gt_inds

    def _filter_invisible(self, bboxes, centers2d, gt_labels, depths, gt_inds=None):
        # filter invisible 2d bboxes
        assert len(bboxes) == len(centers2d) == len(gt_labels) == len(depths)
        if gt_inds is not None:
            assert len(gt_labels) == len(gt_inds)
        fH, fW = self.data_aug_conf["final_dim"]
        indices_maps = np.zeros((fH,fW))
        tmp_bboxes = np.zeros_like(bboxes)
        tmp_bboxes[:, :2] = np.ceil(bboxes[:, :2])
        tmp_bboxes[:, 2:] = np.floor(bboxes[:, 2:])
        tmp_bboxes = tmp_bboxes.astype(np.int64)
        sort_idx = np.argsort(-depths, axis=0, kind='stable')
        tmp_bboxes = tmp_bboxes[sort_idx]
        bboxes = bboxes[sort_idx]
        depths = depths[sort_idx]
        centers2d = centers2d[sort_idx]
        gt_labels = gt_labels[sort_idx]
        if gt_inds is not None:
            gt_inds = gt_inds[sort_idx]
        for i in range(bboxes.shape[0]):
            u1, v1, u2, v2 = tmp_bboxes[i]
            indices_maps[v1:v2, u1:u2] = i
        indices_res = np.unique(indices_maps).astype(np.int64)
        bboxes = bboxes[indices_res]
        depths = depths[indices_res]
        centers2d = centers2d[indices_res]
        gt_labels = gt_labels[indices_res]
        if gt_inds is not None:
            gt_inds = gt_inds[indices_res]

        return bboxes, centers2d, gt_labels, depths, gt_inds

    def _get_rot(self, h):
        return torch.Tensor(
            [
                [np.cos(h), np.sin(h)],
                [-np.sin(h), np.cos(h)],
            ]
        )

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
        else:
            # resize = max(fH / H, fW / W)
            resize = np.mean(self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

@PIPELINES.register_module()
class GlobalRotScaleTransImage():
    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        results["gt_bboxes_3d"].rotate(
            np.array(rot_angle)
        )  

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)

        return results

    def _trans_xyz(self, results, trans):
        trans_mat = torch.eye(4, 4)
        trans_mat[:3, -1] = torch.from_numpy(trans).reshape(1, 3)
        trans_mat_inv = torch.inverse(trans_mat)
        num_view = len(results["lidar2img"])

        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ trans_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ trans_mat_inv).numpy()

    def _rotate_bev_along_z(self, results, angle):
        rot_cos = torch.cos(torch.tensor(angle))
        rot_sin = torch.sin(torch.tensor(angle))

        rot_mat = torch.tensor([[rot_cos, rot_sin, 0, 0], [-rot_sin, rot_cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        rot_mat_inv = torch.inverse(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()

    def _scale_xyz(self, results, scale_ratio):
        scale_mat = torch.tensor(
            [
                [scale_ratio, 0, 0, 0],
                [0, scale_ratio, 0, 0],
                [0, 0, scale_ratio, 0],
                [0, 0, 0, 1],
            ]
        )

        scale_mat_inv = torch.inverse(scale_mat)


        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ scale_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ scale_mat_inv).numpy()


@PIPELINES.register_module()
class BEVGlobalRotScaleTrans(GlobalRotScaleTransImage):
    def __call__(self, results):
        # random rotate
        translation_std = np.array(self.translation_std, dtype=np.float32)

        rot_angle = np.random.uniform(*self.rot_range)
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        trans = np.random.normal(scale=translation_std, size=3).T

        points = results['points']

        self._rotate_bev_along_z(results, rot_angle)
        if self.reverse_angle:
            rot_angle = rot_angle * -1
        points, _ = results["gt_bboxes_3d"].rotate(
            np.array(rot_angle), points
        )

        # random scale
        self._scale_xyz(results, scale_ratio)
        results["gt_bboxes_3d"].scale(scale_ratio)
        points.scale(scale_ratio)

        #random translate
        self._trans_xyz(results, trans)
        results["gt_bboxes_3d"].translate(trans)
        points.translate(trans)
        results['points'] = points

        return results


@PIPELINES.register_module()
class BEVRandomFlip3D:
    """Compared with `RandomFlip3D`, this class directly records the lidar
    augmentation matrix in the `data`."""

    def __call__(self, results):
        flip_horizontal = np.random.choice([0, 1])
        flip_vertical = np.random.choice([0, 1])

        rotation = np.eye(3)
        if flip_horizontal:
            rotation = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ rotation
            if 'points' in results:
                results['points'].flip('horizontal')
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'].flip('horizontal')

        if flip_vertical:
            rotation = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]) @ rotation
            if 'points' in results:
                results['points'].flip('vertical')
            if 'gt_bboxes_3d' in results:
                results['gt_bboxes_3d'].flip('vertical')

        rot_mat = np.eye(4)
        rot_mat[:3, :3] = rotation
        rot_mat = torch.from_numpy(rot_mat).float()
        rot_mat_inv = torch.inverse(rot_mat)
        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = (torch.tensor(results["lidar2img"][view]).float() @ rot_mat_inv).numpy()
            results["extrinsics"][view] = (torch.tensor(results["extrinsics"][view]).float() @ rot_mat_inv).numpy()
        return results

from mmdet3d.datasets.pipelines import Compose
from typing import Dict, Optional

@PIPELINES.register_module()
class RandomChoice(object):
    """Process data with a randomly chosen transform from given candidates.

    Args:
        transforms (list[list]): A list of transform candidates, each is a
            sequence of transforms.
        prob (list[float], optional): The probabilities associated
            with each pipeline. The length should be equal to the pipeline
            number and the sum should be 1. If not given, a uniform
            distribution will be assumed.

    Examples:
        >>> # config
        >>> pipeline = [
        >>>     dict(type='RandomChoice',
        >>>         transforms=[
        >>>             [dict(type='RandomHorizontalFlip')],  # subpipeline 1
        >>>             [dict(type='RandomRotate')],  # subpipeline 2
        >>>         ]
        >>>     )
        >>> ]
    """

    def __init__(self,
                 transforms,
                 prob,
                 stop_epoch=None):

        super().__init__()

        if prob is not None:
            assert len(transforms) == len(prob), \
                '``transforms`` and ``prob`` must have same lengths. ' \
                f'Got {len(transforms)} vs {len(prob)}.'
            assert sum(prob) == 1

        self.prob = prob
        self.transforms = [Compose(transform) for transform in transforms]
        self.stop_epoch = stop_epoch

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        return iter(self.transforms)

    # @cache_randomness
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        if self.stop_epoch is not None and self.epoch >= self.stop_epoch:
            return results

        idx = self.random_pipeline_index()
        return self.transforms[idx](results)

    def __call__(self, results: Dict):

        return self.transform(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str

@PIPELINES.register_module()
class ReplaceImagePath:
    def __init__(self, path):
        # assume path points to data/nuscenes/samples
        cams = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'
        ]
        self.path_root = path
        self.path_list = []
        
        for cam in cams:
            self.path_list.extend(os.listdir(os.path.join(path, cam)))
        
        self.path_list = set(self.path_list)

    def __call__(self, results):
        
        img_paths = results['img_filename']

        for i in range(len(img_paths)):
            filename = img_paths[i].split('/')[-1]
            if filename in self.path_list:
                print("replace", filename)
                img_paths[i] = os.path.join(self.path_root, filename)

        results['img_filename'] = img_paths

        return results
    
@PIPELINES.register_module()
class ModalMask3D(object):

    def __init__(self, mode='test', mask_modal='image', **kwargs):
        super(ModalMask3D, self).__init__()
        self.mode = mode
        self.mask_modal = mask_modal

    def _zero_points(self, points):
        """Zero out points, handling various formats (LiDARPoints, DataContainer, list, tensor)."""
        from mmcv.parallel import DataContainer
        
        if isinstance(points, list):
            for p in points:
                self._zero_single_points(p)
        else:
            self._zero_single_points(points)
        return points
    
    def _zero_single_points(self, p):
        """Zero out a single points object."""
        from mmcv.parallel import DataContainer
        
        if isinstance(p, DataContainer):
            # DataContainer wraps the actual data in .data attribute
            data = p.data
            if hasattr(data, 'tensor'):
                data.tensor = data.tensor * 0.0
            elif hasattr(data, 'mul_'):
                data.mul_(0.0)
            else:
                # numpy array or similar
                data *= 0.0
        elif hasattr(p, 'tensor'):
            p.tensor = p.tensor * 0.0
        elif hasattr(p, 'mul_'):
            p.mul_(0.0)
        else:
            p *= 0.0

    def _zero_images(self, imgs):
        """Zero out images, handling various formats (list, DataContainer, tensor, array)."""
        from mmcv.parallel import DataContainer
        import torch
        
        if isinstance(imgs, DataContainer):
            # Single DataContainer wrapping data - modify in place
            self._zero_single_image_inplace(imgs._data)
            return imgs
        elif isinstance(imgs, list):
            # List of images (or DataContainers)
            for i, img in enumerate(imgs):
                if isinstance(img, DataContainer):
                    self._zero_single_image_inplace(img._data)
                else:
                    imgs[i] = self._zero_single_image(img)
            return imgs
        else:
            return self._zero_single_image(imgs)
    
    def _zero_single_image_inplace(self, img):
        """Zero out a single image in place."""
        import torch
        
        if isinstance(img, torch.Tensor):
            img.zero_()
        elif isinstance(img, np.ndarray):
            img.fill(0)
        elif isinstance(img, list):
            for item in img:
                self._zero_single_image_inplace(item)
    
    def _zero_single_image(self, img):
        """Zero out a single image, returning new object."""
        import torch
        
        if isinstance(img, torch.Tensor):
            return img * 0.0
        elif isinstance(img, np.ndarray):
            return img * 0.0
        else:
            # Try multiplication anyway
            return img * 0.0

    def __call__(self, input_dict):
        if self.mode == 'test':
            if self.mask_modal == 'image':
                input_dict['img'] = self._zero_images(input_dict['img'])
            if self.mask_modal == 'points':
                input_dict['points'] = self._zero_points(input_dict['points'])
        else:
            seed = np.random.rand()
            if seed > 0.75:
                input_dict['img'] = self._zero_images(input_dict['img'])
            elif seed > 0.5:
                input_dict['points'] = self._zero_points(input_dict['points'])

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str

import copy
import mmcv
import inspect
import random
import cv2

try:
    import albumentations
    from albumentations import Compose as AlbumentationsCompose
except ImportError:
    albumentations = None
    AlbumentationsCompose = None

# cv2, numpy, copy, inspect, mmcv are already imported or available in the environment
# Ensure albumentations is checked appropriately

@PIPELINES.register_module()
class ObjectAug2D:
    """Apply a randomly selected Albumentation transform to each 2D object.

    This class iterates through images (single or multi-view) and for each
    bounding box, it crops the object, applies one randomly chosen augmentation
    from the provided list to the crop, and pastes the augmented crop back.
    The bounding box coordinates themselves are not modified.

    Args:
        transforms (list[dict]): A list of Albumentation transform
            configurations. Each dictionary should specify an Albumentation
            transform.
        prob (float): Probability of applying augmentation to each object. Defaults to 1.0.
    """

    def __init__(self, transforms, prob=1.0):
        if albumentations is None:
            raise RuntimeError(
                'albumentations is not installed, which is required for ObjectAug2D'
            )
        # Store a deep copy of the transform configurations
        self.transforms_cfg = copy.deepcopy(transforms)
        self.prob = prob # Add probability attribute

    def _albu_builder(self, cfg):
        """Builds an Albumentation transform object from a configuration dict.
        This method is adapted from the Albu class's albu_builder.
        It handles nested transform structures like OneOf, SomeOf, etc.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed Albumentation transform object.
        """
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()  # Work on a copy as 'pop' modifies the dict

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            # Ensured albumentations is not None in __init__
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            # This case is if the type is already a class object,
            # though typically it's a string from config.
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        # Recursively build nested transforms (e.g., for OneOf, Compose)
        if 'transforms' in args and isinstance(args.get('transforms'), list):
            args['transforms'] = [
                self._albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        """
        Apply augmentations to objects within images.

        Args:
            results (dict): The input data dictionary. Expected keys:
                'img': A single image (np.ndarray) or a list of images.
                'gt_bboxes' (optional): Ground truth bounding boxes.
                    If 'img' is single, 'gt_bboxes' is an np.ndarray.
                    If 'img' is a list, 'gt_bboxes' should be a list of
                    np.ndarrays corresponding to each image.

        Returns:
            dict: The results dictionary with augmented images.
        """
        if not self.transforms_cfg:
            return results  # No transforms configured

        input_imgs = results.get('img')
        if input_imgs is None:
            # Optionally log a warning if 'img' is missing
            return results

        # Standardize image and bbox inputs to lists
        if isinstance(input_imgs, np.ndarray):
            imgs_list = [input_imgs]
            gt_bboxes_list = [results.get('gt_bboxes', np.array([]))]
        elif isinstance(input_imgs, list):
            imgs_list = input_imgs
            gt_bboxes_input = results.get('gt_bboxes', [])
            if (isinstance(gt_bboxes_input, list) and
                    all(isinstance(b, np.ndarray) for b in gt_bboxes_input) and
                    len(gt_bboxes_input) == len(imgs_list)):
                gt_bboxes_list = gt_bboxes_input
            elif isinstance(gt_bboxes_input, np.ndarray) and len(imgs_list) == 1:
                 gt_bboxes_list = [gt_bboxes_input] # Single image in list, single bbox array
            else: # Default to empty bboxes if format is mismatched or missing
                gt_bboxes_list = [np.array([]) for _ in imgs_list]
        else:
            # Optionally log a warning for unexpected 'img' type
            return results

        processed_imgs = []
        for img_idx, single_img_original in enumerate(imgs_list):
            # Work on a copy of the image to avoid modifying the original data in `results`
            # if it's a view or shared.
            img_to_process = single_img_original.copy()
            img_h, img_w = img_to_process.shape[:2]

            if img_idx >= len(gt_bboxes_list):
                processed_imgs.append(img_to_process)
                continue
            
            current_gt_bboxes = gt_bboxes_list[img_idx]

            if not isinstance(current_gt_bboxes, np.ndarray) or current_gt_bboxes.size == 0:
                processed_imgs.append(img_to_process)
                continue

            for bbox in current_gt_bboxes:
                # Add probability check like in ObjectAug3D
                if random.random() > self.prob:
                    continue

                if bbox.shape[0] < 4: # Ensure bbox has at least x1,y1,x2,y2
                    continue
                
                # Convert to int and clip coordinates for safe slicing
                x1, y1, x2, y2 = map(int, bbox[:4])
                y_start = np.clip(y1, 0, img_h)
                y_end = np.clip(y2, 0, img_h)
                x_start = np.clip(x1, 0, img_w)
                x_end = np.clip(x2, 0, img_w)

                if x_start >= x_end or y_start >= y_end:
                    continue  # Skip invalid or zero-area crops

                obj_crop = img_to_process[y_start:y_end, x_start:x_end]
                if obj_crop.size == 0:
                    continue
                
                original_dtype = obj_crop.dtype
                crop_for_albu = None

                if original_dtype == np.float32:
                    # Assuming float32 images are in [0, 255] range from LoadMultiViewImageFromFiles.
                    # Normalize to [0, 1] for albumentations.
                    # Also clip before division to prevent potential warnings/errors if values are slightly outside 0-255
                    crop_for_albu = np.clip(obj_crop, 0, 255) / 255.0 
                elif original_dtype == np.uint8:
                    # Albumentations will expect uint8 in [0,255] and output uint8 in [0,255]
                    crop_for_albu = obj_crop.copy() # Needs to be a copy for augmentation
                else:
                    print(f"Warning: Unexpected dtype {original_dtype} for ObjectAug2D input crop. Attempting to convert to uint8.")
                    try:
                        # Fallback: try to convert to uint8 [0,255]
                        crop_for_albu = np.clip(obj_crop.astype(np.float32), 0, 255).astype(np.uint8)
                    except Exception as e_conv:
                        print(f"Error converting crop to uint8: {e_conv}. Skipping augmentation for this bbox.")
                        continue
                
                # Randomly select one transform configuration
                chosen_transform_cfg = random.choice(self.transforms_cfg)
                
                try:
                    # Build the transform object. Pass a deepcopy of the config
                    # as the builder might modify it (e.g. args.pop).
                    transform_obj = self._albu_builder(copy.deepcopy(chosen_transform_cfg))
                    
                    # Apply augmentation
                    # The input to albumentations is either uint8 [0,255] or float32 [0,1]
                    augmented_data = transform_obj(image=crop_for_albu)
                    augmented_crop_internal = augmented_data['image'] # dtype will match crop_for_albu (input to transform_obj)
                    
                    # Denormalize if original was float32 and augmentation input was float32 [0,1]
                    if original_dtype == np.float32 and crop_for_albu.dtype == np.float32:
                        # augmented_crop_internal is float32 [0,1] (output from albumentations)
                        augmented_crop_processed = augmented_crop_internal * 255.0
                        # Clip to ensure it's within [0, 255] after denormalization,
                        # as some augs might slightly exceed [0,1] without explicit clipping in their definition.
                        augmented_crop_processed = np.clip(augmented_crop_processed, 0, 255)
                    elif original_dtype == np.uint8 and augmented_crop_internal.dtype == np.uint8:
                        # original was uint8, augmented_crop_internal is uint8 [0,255]
                        augmented_crop_processed = augmented_crop_internal
                    elif crop_for_albu.dtype == np.uint8 and augmented_crop_internal.dtype == np.uint8: # Fallback case produced uint8
                         augmented_crop_processed = augmented_crop_internal
                         # If original_dtype was float, we might want to convert back,
                         # but paste operation below will handle astype(original_dtype)
                    else:
                        # This case should ideally not be hit if logic above is correct,
                        # but as a safeguard, assume internal format matches original if not explicitly handled
                        print(f"Warning: Unhandled dtype combination. Original: {original_dtype}, Albu_input: {crop_for_albu.dtype}, Albu_output: {augmented_crop_internal.dtype}. Using output directly.")
                        augmented_crop_processed = augmented_crop_internal

                    # Ensure augmented crop is resized to original crop dimensions
                    orig_crop_h, orig_crop_w = y_end - y_start, x_end - x_start
                    if augmented_crop_processed.shape[0] != orig_crop_h or \
                       augmented_crop_processed.shape[1] != orig_crop_w:
                        augmented_crop_processed = cv2.resize(augmented_crop_processed,
                                                    (orig_crop_w, orig_crop_h),
                                                    interpolation=cv2.INTER_LINEAR)
                                                    
                    img_to_process[y_start:y_end, x_start:x_end] = augmented_crop_processed.astype(original_dtype, copy=False)
                except Exception as e:
                    # Optionally log the error:
                    print(f"Error applying transform {chosen_transform_cfg.get('type')} to bbox ({x_start},{y_start},{x_end},{y_end}): {e}")
                    # If augmentation fails, the original crop content remains unchanged in img_to_process.
                    pass
            
            processed_imgs.append(img_to_process)

        # Update the 'img' field in results
        if isinstance(results.get('img'), np.ndarray) and len(processed_imgs) == 1:
            results['img'] = processed_imgs[0]
        elif isinstance(results.get('img'), list) and len(processed_imgs) == len(results['img']):
            results['img'] = processed_imgs
        # If processed_imgs structure doesn't match original, original 'img' in results is returned (implicitly).
        # This can happen if initial checks fail or img_list/bbox_list structure is highly unusual.

        return results

    def __repr__(self):
        """String representation of the transform."""
        return f"{self.__class__.__name__}(transforms={self.transforms_cfg}, prob={self.prob})"


@PIPELINES.register_module()
class SimpleAlbu:
    """A simplified version of Albu for image-only augmentations, supporting multi-view images.

    This class applies a composed Albumentation pipeline to each image.
    If the input 'img' is a list of images (multi-view), each image is augmented
    independently.

    Args:
        transforms (list[dict]): A list of Albumentation transform configurations.
    """
    def __init__(self, transforms):
        if albumentations is None:
            raise RuntimeError(
                'albumentations is not installed, which is required for SimpleAlbu'
            )
        if not isinstance(transforms, list):
            raise TypeError('transforms must be a list of dicts')
        
        self.transforms_cfg = copy.deepcopy(transforms)
        # Build the main augmentation pipeline from the list of transform configs
        # The Compose itself will always be attempted (p=1.0 by default),
        # individual transform probabilities within the list are handled by albumentations.
        self.aug = AlbumentationsCompose([self._albu_builder(t) for t in self.transforms_cfg])

    def _albu_builder(self, cfg):
        """Builds an Albumentation transform object from a configuration dict.
        (Adapted from the original Albu class / ObjectAug2D)

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".

        Returns:
            obj: The constructed Albumentation transform object.
        """
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()
        obj_type = args.pop('type')

        if mmcv.is_str(obj_type):
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid class, but got {type(obj_type)}')

        if 'transforms' in args and isinstance(args.get('transforms'), list):
            # Handle nested transforms for compositions like OneOf, SomeOf, etc.
            args['transforms'] = [
                self._albu_builder(transform)
                for transform in args['transforms' ]
            ]
        return obj_cls(**args)

    def _apply_aug_to_single_image(self, img_array):
        """Prepares, augments, and post-processes a single image array."""
        if not isinstance(img_array, np.ndarray):
            print(f"Warning (SimpleAlbu): Input item is not a numpy array (type: {type(img_array)}). Returning as is.")
            return img_array

        original_dtype = img_array.dtype
        img_for_albu = None

        if original_dtype == np.float32:
            # Assuming float32 images from loader are in [0, 255] range.
            # Normalize to [0, 1] for Albumentations.
            img_for_albu = np.clip(img_array, 0.0, 255.0) / 255.0
        elif original_dtype == np.uint8:
            # Albumentations handles uint8 [0,255] directly.
            img_for_albu = img_array # Still pass a copy if original might be a view and aug is in-place
        else:
            print(f"Warning (SimpleAlbu): Unexpected dtype {original_dtype}. Attempting conversion to float32 [0,1].")
            try:
                img_for_albu = np.clip(img_array.astype(np.float32), 0.0, 255.0) / 255.0
            except Exception as e:
                print(f"Error converting image of dtype {original_dtype} to float32: {e}. Returning original.")
                return img_array
        
        # Ensure img_for_albu is C-contiguous if it's a copy, some albumentations might need it.
        # However, most common ops handle non-contiguous well. Let's rely on albumentations.
        # if img_for_albu is not img_array and not img_for_albu.flags.c_contiguous:
        #    img_for_albu = np.ascontiguousarray(img_for_albu)

        try:
            augmented_data = self.aug(image=img_for_albu)
            augmented_img_internal = augmented_data['image'] # dtype should match img_for_albu
        except Exception as e:
            print(f"Error during SimpleAlbu augmentation: {e}. Returning original image.")
            # If error occurs, return the original image array converted back to its original dtype
            # This handles cases where img_for_albu was a normalized version.
            if original_dtype == np.float32 and img_for_albu.dtype == np.float32 and img_array.dtype == np.float32:
                 # img_array was [0,255], img_for_albu was [0,1], error happened. Return original img_array.
                 return img_array
            elif original_dtype == np.uint8 and img_for_albu.dtype == np.uint8:
                 return img_array # Original was uint8, return it.
            else: # Fallback, try to return original img_array
                 return img_array

        # Post-process: Denormalize if original was float32, and cast back to original dtype.
        final_augmented_img = None
        current_augmented_dtype = augmented_img_internal.dtype

        if original_dtype == np.float32:
            if current_augmented_dtype == np.float32: # Expected path: input was [0,1] float, output is [0,1] float
                final_augmented_img = np.clip(augmented_img_internal * 255.0, 0.0, 255.0)
            else: # Should not happen if albumentations behaves consistently with float input
                print(f"Warning (SimpleAlbu): Augmented image dtype {current_augmented_dtype} mismatches expected float32 after float32 input. Attempting conversion.")
                final_augmented_img = np.clip(augmented_img_internal.astype(np.float32) * 255.0, 0.0, 255.0)
            final_augmented_img = final_augmented_img.astype(original_dtype)
        elif original_dtype == np.uint8:
            if current_augmented_dtype == np.uint8: # Expected path: input was uint8, output is uint8
                final_augmented_img = augmented_img_internal
            else: # Augmentation changed uint8 to float (e.g. some normalization inside a transform)
                print(f"Warning (SimpleAlbu): Augmented image dtype {current_augmented_dtype} is not uint8 after uint8 input. Clipping and casting.")
                final_augmented_img = np.clip(augmented_img_internal, 0, 255).astype(np.uint8)
            # Ensure it's actually uint8
            final_augmented_img = final_augmented_img.astype(np.uint8)
        else: # Original dtype was unusual, try to restore it after float32 [0,1] processing
            print(f"Warning (SimpleAlbu): Restoring unusual original dtype {original_dtype} after processing.")
            processed_float = augmented_img_internal.astype(np.float32)
            if img_for_albu.dtype == np.float32 : # if it was normalized from original
                processed_float = np.clip(processed_float * 255.0, 0.0, 255.0)
            final_augmented_img = processed_float.astype(original_dtype)
        
        return final_augmented_img

    def __call__(self, results):
        if not self.transforms_cfg:
            return results

        input_imgs_data = results.get('img')
        if input_imgs_data is None:
            print("Warning (SimpleAlbu): 'img' key not found in results or is None.")
            return results

        if isinstance(input_imgs_data, np.ndarray):  # Single image
            results['img'] = self._apply_aug_to_single_image(input_imgs_data)
        elif isinstance(input_imgs_data, list):  # List of images (multi-view)
            augmented_imgs_list = []
            for single_img in input_imgs_data:
                augmented_imgs_list.append(self._apply_aug_to_single_image(single_img))
            results['img'] = augmented_imgs_list
        else:
            print(f"Warning (SimpleAlbu): 'img' key in results is of unexpected type: {type(input_imgs_data)}. Expected np.ndarray or list. No augmentation applied.")
            # No change to results['img'] if type is not recognized

        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(transforms={self.transforms_cfg})"

@PIPELINES.register_module()
class ObjectAug3D:
    """Randomly apply one augmentation transform to the point cloud of each 3D object.

    This class iterates over all 3D bounding boxes and applies a randomly selected augmentation to points inside each box.

    Args:
        transforms (list[dict]): List of augmentation transform configs. Each dict should specify one augmentation method and its parameters.
        prob (float): Probability of applying augmentation to each object. Defaults to 1.0.
    """

    def __init__(self, transforms, prob=1.0):
        self.transforms_cfg = copy.deepcopy(transforms)
        self.prob = prob
        # Supported augmentation methods
        self.aug_methods = {
            'PointDownsample': self._point_downsample,
            'SweepDownsample': self._sweep_downsample,
            'GaussianNoise': self._gaussian_noise,
            'DropPart': self._drop_part
        }

    def _point_downsample(self, points, **kwargs):
        """Random point dropout

        Args:
            points (np.ndarray): Point cloud array with shape (N, >=3)
            keep_ratio (float): Ratio of points to keep. Defaults to 0.8.
        
        Returns:
            np.ndarray: Augmented point cloud
        """
        keep_ratio = kwargs.get('keep_ratio', 0.8)
        num_points = points.shape[0]
        keep_num = max(int(num_points * keep_ratio), 1)  # Keep at least one point
        
        if num_points <= 1:
            return points
        
        indices = np.random.choice(num_points, keep_num, replace=False)
        return points[indices]

    def _sweep_downsample(self, points, **kwargs):
        """Random LiDAR sweep dropout
        
        Assume the last column of the point cloud is the timestamp representing different sweeps.
        Args:
            points (np.ndarray): Point cloud array with shape (N, >=4), assuming the last column is the sweep id or timestamp.
            keep_ratio (float): Ratio of sweeps to keep. Defaults to 0.8.
        
        Returns:
            np.ndarray: Augmented point cloud
        """
        keep_ratio = kwargs.get('keep_ratio', 0.8)
        
        # Ensure the point cloud has enough dimensions to include timestamp information.
        if points.shape[1] < 4:
            return points  # Return the original point cloud directly if there is no timestamp dimension.
        
        # Get the timestamp column, usually the last column.
        time_column = points[:, -1]
        # Get unique timestamps / sweeps.
        unique_times = np.unique(time_column)
        
        if len(unique_times) <= 1:
            return points
        
        # Randomly select sweeps to keep.
        keep_num = max(int(len(unique_times) * keep_ratio), 1)
        keep_times = np.random.choice(unique_times, keep_num, replace=False)
        
        # Keep the selected sweeps.
        keep_mask = np.isin(time_column, keep_times)
        return points[keep_mask]

    def _gaussian_noise(self, points, **kwargs):
        """Add Gaussian-noise points
        
        Args:
            points (np.ndarray): Point cloud array with shape (N, >=3)
            noise_std (float): Standard deviation of Gaussian noise. Defaults to 0.02.
            noise_ratio (float): Ratio of noise points to add. Defaults to 0.5.
        
        Returns:
            np.ndarray: Augmented point cloud
        """
        noise_std = kwargs.get('noise_std', 0.02)
        noise_ratio = kwargs.get('noise_ratio', 0.5)
        
        num_points = points.shape[0]
        num_features = points.shape[1]
        noise_num = max(int(num_points * noise_ratio), 1)
        
        # Create noise points.
        noise_points = points[np.random.choice(num_points, noise_num, replace=True)]
        # Add Gaussian noise to the first 3 dimensions (xyz).
        noise = np.random.normal(0, noise_std, (noise_num, 3))
        noise_points[:, :3] += noise
        
        # Merge original points and noise points.
        augmented_points = np.vstack([points, noise_points])
        return augmented_points

    def _drop_part(self, points, **kwargs):
        """Divide the bbox into an n x n grid and randomly drop points from m grid cells.
        
        Args:
            points (np.ndarray): Point cloud array with shape (N, >=3)
            grid_size (int): Grid size n. Defaults to 2.
            drop_num (int): Number of grid cells m to drop. Defaults to 1.
        
        Returns:
            np.ndarray: Augmented point cloud
        """
        grid_size = kwargs.get('grid_size', 2)
        drop_num = kwargs.get('drop_num', 1)
        
        if points.shape[0] < 1:
            return points
        
        # Compute point cloud bounds.
        x_min, y_min = np.min(points[:, :2], axis=0)
        x_max, y_max = np.max(points[:, :2], axis=0)
        
        # Compute each grid cell size.
        x_step = (x_max - x_min) / grid_size
        y_step = (y_max - y_min) / grid_size
        
        # Very small grid cells may cause issues.
        if x_step < 1e-6 or y_step < 1e-6:
            return points
        
        # Generate all possible grid indices.
        grids = [(i, j) for i in range(grid_size) for j in range(grid_size)]
        
        # Randomly select grid cells to drop.
        drop_num = min(drop_num, len(grids))
        drop_grids = random.sample(grids, drop_num)
        
        # Create a mask indicating points to keep.
        keep_mask = np.ones(points.shape[0], dtype=bool)
        
        for i, j in drop_grids:
            # Compute grid cell bounds.
            grid_x_min = x_min + i * x_step
            grid_x_max = x_min + (i + 1) * x_step
            grid_y_min = y_min + j * y_step
            grid_y_max = y_min + (j + 1) * y_step
            
            # Find points inside this grid cell.
            in_grid = ((points[:, 0] >= grid_x_min) & (points[:, 0] < grid_x_max) &
                       (points[:, 1] >= grid_y_min) & (points[:, 1] < grid_y_max))
            
            # Mark these points as dropped.
            keep_mask = keep_mask & ~in_grid
        
        return points[keep_mask]

    def __call__(self, results):
        """
        Apply augmentation to 3D object point clouds.

        Args:
            results (dict): Input data dict. Expected keys:
                'points': Point cloud data, as np.ndarray.
                'gt_bboxes_3d': 3D bounding boxes.

        Returns:
            dict: Result dict containing the augmented point cloud.
        """
        if not self.transforms_cfg:
            return results  # No transform is configured.
        
        # Check whether required inputs exist.
        if 'points' not in results or 'gt_bboxes_3d' not in results:
            return results
        
        points = results['points'].tensor.numpy() if hasattr(results['points'], 'tensor') else results['points']
        gt_bboxes_3d = results['gt_bboxes_3d'].tensor.numpy() if hasattr(results['gt_bboxes_3d'], 'tensor') else results['gt_bboxes_3d']
        
        # Ensure point cloud and bbox data are valid.
        if not isinstance(points, np.ndarray) or not isinstance(gt_bboxes_3d, np.ndarray):
            return results
        
        # Return directly if there are no bounding boxes.
        if gt_bboxes_3d.size == 0:
            return results
        
        # Deep-copy point cloud data to avoid mutating the original data.
        points_augmented = points.copy()
        
        # Iterate over each 3D bounding box.
        for box_idx, box in enumerate(gt_bboxes_3d):
            # Apply augmentation only when the random value is below the probability threshold.
            if random.random() > self.prob:
                continue
            
            # Extract 3D bounding box parameters.
            if len(box) >= 7:  # Assume format is (x, y, z, l, w, h, yaw).
                # x, y, z, l, w, h, yaw = box[:7] # Old: assume z is the center.
                x, y, z_bottom, l, w, h, yaw = box[:7] # New: explicitly treat z as the bottom center.
                box_center_z = z_bottom + h / 2 # Compute geometric center z.
                
                # Extract points according to the 3D box.
                # Use a simplified method here: only consider the unrotated bounding box.
                # In real use, a rotation matrix may need to be considered.
                
                # Inverse rotation matrix, mapping rotated coordinates back to the original frame.
                cos_yaw = np.cos(-yaw)
                sin_yaw = np.sin(-yaw)
                rot_matrix = np.array([
                    [cos_yaw, -sin_yaw, 0],
                    [sin_yaw, cos_yaw, 0],
                    [0, 0, 1]
                ])
                
                # Transform points into a coordinate frame centered at the object geometric center.
                # centered_points = points_augmented[:, :3] - np.array([x, y, z]) # Old: subtract the assumed center.
                centered_points = points_augmented[:, :3] - np.array([x, y, box_center_z]) # New: subtract the geometric center.
                rotated_points = np.dot(centered_points, rot_matrix.T)
                
                # Find points inside the bounding box, relative to the geometric center.
                mask_in_box = ((rotated_points[:, 0] >= -l/2) & (rotated_points[:, 0] <= l/2) &
                               (rotated_points[:, 1] >= -w/2) & (rotated_points[:, 1] <= w/2) &
                               (rotated_points[:, 2] >= -h/2) & (rotated_points[:, 2] <= h/2))
                
                # Continue to the next box if there are no points inside this box.
                if not np.any(mask_in_box):
                    continue
                
                # Randomly select one augmentation method.
                chosen_transform_cfg = random.choice(self.transforms_cfg)
                transform_type = chosen_transform_cfg.get('type')
                
                if transform_type in self.aug_methods:
                    try:
                        # Extract points inside the bounding box.
                        points_in_box = points_augmented[mask_in_box]
                        
                        # Apply the selected augmentation method.
                        augmented_points = self.aug_methods[transform_type](
                            points_in_box, **{k: v for k, v in chosen_transform_cfg.items() if k != 'type'})
                        
                        # New logic: remove points inside the box first, then add augmented points.
                        points_outside_box = points_augmented[~mask_in_box]
                        points_augmented = np.vstack([points_outside_box, augmented_points])

                    except Exception as e:
                        # Optional: log the error.
                        print(f"Error applying transform {transform_type} to bounding box {box_idx} : {e}")

        
        # Update the point cloud in the result dict.
        if hasattr(results['points'], 'tensor'):
            results['points'] = results['points'].new_point(points_augmented)
        else:
            results['points'] = points_augmented
        
        return results

    def __repr__(self):
        """Return the string representation of this module."""
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms_cfg}, '
        repr_str += f'prob={self.prob})'
        return repr_str
    
@PIPELINES.register_module()
class MultiModalRandMask(object):
    """Multi-modal Random Mask augmentation similar to MAE.
    
    Randomly masks patches of the image/lidar by dividing into grids and 
    randomly dropping a percentage of them.
    
    Args:
        grid_size (int or list/tuple): Size of each grid patch. 
            - If int, uses fixed square patches of that size.
            - If list/tuple of length 2 [min, max], randomly samples grid size in that range.
        mask_ratio (float or list/tuple): Ratio of patches to mask.
            - If float, uses fixed mask ratio (0.0 to 1.0).
            - If list/tuple of length 2 [min, max], randomly samples mask ratio in that range.
        consistent (bool): If True, apply same mask pattern to image and lidar.
        prob (float): Probability of applying this augmentation.
        fixed_prob (bool): If True, keep prob constant during training.
        max_iter (int): Maximum iterations for prob scheduling.
        aug_lidar_prob (float): Probability of augmenting lidar (vs image).
        aug_both (bool): If True, always augment image and optionally lidar.
        filter_2d_gt (bool): If True, filter heavily occluded 2D GT boxes.
        occlusion_threshold (float): Threshold for filtering 2D boxes (default 0.2 = 80% occlusion).
    """
    
    def __init__(
        self,
        grid_size=32,
        mask_ratio=0.5,
        consistent=True,
        prob=1.0,
        fixed_prob=False,
        max_iter=0,
        aug_lidar_prob=0.5,
        aug_both=False,
        filter_2d_gt=False,
        occlusion_threshold=0.2,
    ):
        # Parse grid_size: can be fixed int or range [min, max]
        if isinstance(grid_size, (list, tuple)):
            assert len(grid_size) == 2, "grid_size range must be [min, max]"
            self.grid_size_range = grid_size
            self.grid_size_fixed = None
        else:
            self.grid_size_range = None
            self.grid_size_fixed = grid_size
        
        # Parse mask_ratio: can be fixed float or range [min, max]
        if isinstance(mask_ratio, (list, tuple)):
            assert len(mask_ratio) == 2, "mask_ratio range must be [min, max]"
            assert 0.0 <= mask_ratio[0] <= 1.0 and 0.0 <= mask_ratio[1] <= 1.0, "mask_ratio must be in [0.0, 1.0]"
            self.mask_ratio_range = mask_ratio
            self.mask_ratio_fixed = None
        else:
            assert 0.0 <= mask_ratio <= 1.0, "mask_ratio must be in [0.0, 1.0]"
            self.mask_ratio_range = None
            self.mask_ratio_fixed = mask_ratio
        self.consistent = consistent
        self.st_prob = prob
        self.prob = prob
        self.fixed_prob = fixed_prob
        self.iter = None
        self.max_iter = max_iter
        self.aug_lidar_prob = aug_lidar_prob
        self.aug_both = aug_both
        self.filter_2d_gt = filter_2d_gt
        self.occlusion_threshold = occlusion_threshold

    def set_iter(self, iter):
        self.iter = iter
        if not self.fixed_prob:
            self.set_prob()

    def set_prob(self):
        self.prob = self.st_prob * self.iter / self.max_iter

    def _generate_random_mask(self, h, w):
        """Generate random mask similar to MAE.
        
        Args:
            h (int): Image height
            w (int): Image width
            
        Returns:
            np.ndarray: Binary mask of shape (h, w) with 1 for keep, 0 for drop
        """
        # Sample grid size if range is provided
        if self.grid_size_range is not None:
            grid_size = np.random.randint(self.grid_size_range[0], self.grid_size_range[1] + 1)
        else:
            grid_size = self.grid_size_fixed
        
        # Sample mask ratio if range is provided
        if self.mask_ratio_range is not None:
            mask_ratio = np.random.uniform(self.mask_ratio_range[0], self.mask_ratio_range[1])
        else:
            mask_ratio = self.mask_ratio_fixed
        
        # Use square grids
        grid_h = grid_w = grid_size
        
        # Calculate number of grids
        n_grids_h = (h + grid_h - 1) // grid_h
        n_grids_w = (w + grid_w - 1) // grid_w
        total_grids = n_grids_h * n_grids_w
        
        # Randomly select grids to mask
        n_masked = int(total_grids * mask_ratio)
        masked_indices = np.random.choice(total_grids, n_masked, replace=False)
        masked_set = set(masked_indices)
        
        # Create mask
        mask = np.ones((h, w), dtype=np.float32)
        
        for idx in range(total_grids):
            if idx in masked_set:
                grid_i = idx // n_grids_w
                grid_j = idx % n_grids_w
                
                h_start = grid_i * grid_h
                h_end = min((grid_i + 1) * grid_h, h)
                w_start = grid_j * grid_w
                w_end = min((grid_j + 1) * grid_w, w)
                
                mask[h_start:h_end, w_start:w_end] = 0
        
        return mask

    def __call__(self, results):
        if np.random.rand() > self.prob:
            return results
        
        if self.aug_both:
            # aug_both mode: always augment images and use aug_lidar_prob to decide whether to also augment LiDAR.
            augment_image = True
            augment_lidar = np.random.rand() < self.aug_lidar_prob
        else:
            # Original mode: choose which modality to augment; True means LiDAR, False means image.
            augment_lidar = np.random.rand() < self.aug_lidar_prob
            augment_image = not augment_lidar
        
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        num_views = len(imgs)
        
        # Generate different random mask for each view
        masks = []
        for view_idx in range(num_views):
            mask = self._generate_random_mask(h, w)
            masks.append(mask)
        
        # Stack masks for easy processing, shape: (num_views, h, w)
        masks_array = np.stack(masks, axis=0)

        gridmask_info = dict(
            keep_masks=None,
            apply_image=bool(augment_image),
            apply_lidar=bool(augment_lidar),
            consistent=self.consistent
        )

        if augment_image or augment_lidar:
            gridmask_info['keep_masks'] = [mask.astype(np.uint8) for mask in masks]

        results['gridmask_info'] = gridmask_info

        if augment_lidar:
            # Augment the LiDAR point cloud.
            coords = results['points'][:, :3].tensor
            lidar2img = coords.new_tensor(np.asarray(results['lidar2img']))

            # lidar2img projection
            points_img = torch.cat([coords, torch.ones_like(coords[:, :1])], dim=1) @ lidar2img.permute(0, 2, 1)
            points_img[..., 2] = torch.clip(points_img[..., 2], min=1e-5, max=1e5)
            points_img[..., 0] /= points_img[..., 2]
            points_img[..., 1] /= points_img[..., 2]

            points = results['points']
            
            # For each view, use its corresponding mask
            mask3d = torch.ones(points.shape[0], dtype=torch.bool)
            for i in range(num_views):
                mask_tensor = points_img.new_tensor(masks[i][:, :, None])  # Add channel dimension
                mask_value = torch.ones(points.shape[0], dtype=torch.bool)
                mask_indices = (points_img[i, :, 0] >= 0) & (points_img[i, :, 0] < w) & \
                               (points_img[i, :, 1] >= 0) & (points_img[i, :, 1] < h)
                mask_value[mask_indices] = mask_tensor[
                    points_img[i, mask_indices, 1].long(), 
                    points_img[i, mask_indices, 0].long()
                ].bool().squeeze()
                
                mask3d = mask3d & mask_value
            
            # If not in consistent mode, use the inverse mask for LiDAR.
            if not self.consistent:
                mask3d = ~mask3d

            points = points[mask3d]
            results.update(points=points)
            
        if augment_image:
            # Augment images; each view uses a different mask.
            augmented_imgs = []
            for view_idx, img in enumerate(imgs):
                mask = masks[view_idx][:, :, None]  # Add channel dimension (H, W, 1)
                augmented_img = img * mask
                augmented_imgs.append(augmented_img)
            results.update(img=augmented_imgs)
            
        if self.filter_2d_gt and augment_image and 'gridmask_info' in results and 'keep_masks' in results['gridmask_info']:
            # --- Filter heavily occluded 2D labels ---
            if 'gt_bboxes' in results and results['gt_bboxes'] is not None:
                new_gt_bboxes = []
                new_gt_labels = []
                new_centers2d = []
                new_depths = []
                
                for view_idx in range(num_views):
                    mask_2d = masks[view_idx]  # Use view-specific mask
                    
                    if len(results['gt_bboxes']) > view_idx:
                        view_gt_bboxes = results['gt_bboxes'][view_idx]
                        view_gt_labels = results.get('gt_labels', [None] * num_views)[view_idx] if 'gt_labels' in results else None
                        view_centers2d = results.get('centers2d', [None] * num_views)[view_idx] if 'centers2d' in results else None
                        view_depths = results.get('depths', [None] * num_views)[view_idx] if 'depths' in results else None
                        
                        if view_gt_bboxes is not None and len(view_gt_bboxes) > 0:
                            keep_indices = []
                            
                            for box_idx, bbox in enumerate(view_gt_bboxes):
                                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                
                                x1 = max(0, min(x1, w - 1))
                                y1 = max(0, min(y1, h - 1))
                                x2 = max(0, min(x2, w - 1))
                                y2 = max(0, min(y2, h - 1))
                                
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                
                                box_mask = mask_2d[y1:y2, x1:x2]
                                if box_mask.size > 0:
                                    visible_ratio = np.mean(box_mask)
                                    if visible_ratio >= self.occlusion_threshold:
                                        keep_indices.append(box_idx)
                            
                            if len(keep_indices) > 0:
                                keep_indices = np.array(keep_indices)
                                filtered_bboxes = view_gt_bboxes[keep_indices] if isinstance(view_gt_bboxes, np.ndarray) else [view_gt_bboxes[i] for i in keep_indices]
                                filtered_labels = view_gt_labels[keep_indices] if view_gt_labels is not None and isinstance(view_gt_labels, np.ndarray) else ([view_gt_labels[i] for i in keep_indices] if view_gt_labels is not None else None)
                                filtered_centers2d = view_centers2d[keep_indices] if view_centers2d is not None and isinstance(view_centers2d, np.ndarray) else ([view_centers2d[i] for i in keep_indices] if view_centers2d is not None else None)
                                filtered_depths = view_depths[keep_indices] if view_depths is not None and isinstance(view_depths, np.ndarray) else ([view_depths[i] for i in keep_indices] if view_depths is not None else None)
                            else:
                                # Keep the original dtype to avoid type mismatches from the default float64.
                                filtered_bboxes = np.array([], dtype=view_gt_bboxes.dtype) if isinstance(view_gt_bboxes, np.ndarray) else []
                                filtered_labels = np.array([], dtype=view_gt_labels.dtype) if view_gt_labels is not None and isinstance(view_gt_labels, np.ndarray) else ([] if view_gt_labels is not None else None)
                                filtered_centers2d = np.array([], dtype=view_centers2d.dtype).reshape(0, 2) if view_centers2d is not None and isinstance(view_centers2d, np.ndarray) else ([] if view_centers2d is not None else None)
                                filtered_depths = np.array([], dtype=view_depths.dtype) if view_depths is not None and isinstance(view_depths, np.ndarray) else ([] if view_depths is not None else None)
                            
                            new_gt_bboxes.append(filtered_bboxes)
                            if view_gt_labels is not None:
                                new_gt_labels.append(filtered_labels)
                            if view_centers2d is not None:
                                new_centers2d.append(filtered_centers2d)
                            if view_depths is not None:
                                new_depths.append(filtered_depths)
                        else:
                            new_gt_bboxes.append(view_gt_bboxes if view_gt_bboxes is not None else np.array([], dtype=np.float32))
                            if view_gt_labels is not None or 'gt_labels' in results:
                                new_gt_labels.append(view_gt_labels if view_gt_labels is not None else np.array([], dtype=np.int64))
                            if view_centers2d is not None or 'centers2d' in results:
                                new_centers2d.append(view_centers2d if view_centers2d is not None else np.array([], dtype=np.float32).reshape(0, 2))
                            if view_depths is not None or 'depths' in results:
                                new_depths.append(view_depths if view_depths is not None else np.array([], dtype=np.float32))
                    else:
                        new_gt_bboxes.append(np.array([], dtype=np.float32))
                        if 'gt_labels' in results:
                            new_gt_labels.append(np.array([], dtype=np.int64))
                        if 'centers2d' in results:
                            new_centers2d.append(np.array([], dtype=np.float32).reshape(0, 2))
                        if 'depths' in results:
                            new_depths.append(np.array([], dtype=np.float32))
                
                results['gt_bboxes'] = new_gt_bboxes
                if 'gt_labels' in results and len(new_gt_labels) > 0:
                    results['gt_labels'] = new_gt_labels
                if 'centers2d' in results and len(new_centers2d) > 0:
                    results['centers2d'] = new_centers2d
                if 'depths' in results and len(new_depths) > 0:
                    results['depths'] = new_depths
        
        # --- GT visibility computation ---
        gt_bboxes = results['gt_bboxes_3d']
        num_gts = len(gt_bboxes.tensor)
        gt_pts_visible_mask = gt_bboxes.tensor.new_ones(num_gts, dtype=torch.bool)
        gt_img_visible_mask = gt_bboxes.tensor.new_ones(num_gts, dtype=torch.bool)

        if augment_lidar:
            final_points = results['points'].tensor
            if num_gts > 0:
                if final_points.shape[0] > 0:
                    points_np = final_points.cpu().numpy()
                    boxes_np = gt_bboxes.tensor.cpu().numpy()
                    
                    point_in_box_mask_np = points_in_rbbox(points_np, boxes_np[:, :7])
                    points_per_box = point_in_box_mask_np.sum(axis=0)
                    gt_pts_visible_mask = torch.from_numpy(points_per_box > 0).to(gt_bboxes.tensor.device, dtype=torch.bool)
                else:
                    gt_pts_visible_mask[:] = False
        
        if augment_image:
            if num_gts > 0:
                corners = gt_bboxes.corners
                lidar2img = results.get('lidar2img')
                if lidar2img is None:
                    return results
                if isinstance(lidar2img, list):
                    lidar2img = np.asarray(lidar2img)
                lidar2img = torch.from_numpy(lidar2img).float().to(corners.device)
                
                # Use view-specific masks for visibility checking
                for i in range(num_gts):
                    gt_is_visible_in_any_view = False
                    gt_corners = corners[i]
                    gt_corners_hom = torch.cat([gt_corners, torch.ones_like(gt_corners[..., :1])], dim=-1)
                    
                    points_img = lidar2img @ gt_corners_hom.T
                    points_img = points_img.permute(0, 2, 1)
                    points_img[..., :2] /= points_img[..., 2:3].clamp(min=1e-5)

                    for cam_idx in range(points_img.shape[0]):
                        # Use the mask specific to this camera view
                        img_mask = torch.from_numpy(masks[cam_idx]).to(corners.device)
                        H, W = img_mask.shape
                        
                        cam_points = points_img[cam_idx]
                        depth = cam_points[:, 2]
                        points_xy = cam_points[:, :2]
                        
                        on_img = (points_xy[:, 0] >= 0) & (points_xy[:, 0] < W) & \
                                 (points_xy[:, 1] >= 0) & (points_xy[:, 1] < H) & \
                                 (depth > 1e-5)
                        
                        if not on_img.any():
                            continue

                        points_on_img = points_xy[on_img].long()
                        mask_values = img_mask[points_on_img[:, 1], points_on_img[:, 0]]
                        
                        if (mask_values == 1).sum() >= 2:
                            gt_is_visible_in_any_view = True
                            break
                    
                    if not gt_is_visible_in_any_view:
                        gt_img_visible_mask[i] = False
        
        results['gt_pts_visible_mask'] = gt_pts_visible_mask
        results['gt_img_visible_mask'] = gt_img_visible_mask

        return results


@PIPELINES.register_module()
class MultiModalGridMask(object):

    def __init__(
        self,
        use_h,
        use_w,
        max_iter=0,
        start_iter=0,
        rotate=1,
        offset=False,
        max_length_ratio=1.0,
        ratio=0.5,
        mode=0,
        consistent=True,
        prob=1.0,
        fixed_prob=False,
        aug_lidar_prob=0.5,
        aug_both=False,
        filter_2d_gt=False, # filter occluded 2D GT boxes
        skip_scenes=None
    ):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.max_length_ratio = max_length_ratio
        self.mode = mode
        self.consistent = consistent
        self.st_prob = prob
        self.prob = prob
        self.iter = None
        self.max_iter = max_iter
        self.start_iter = start_iter
        self.fixed_prob = fixed_prob
        self.aug_lidar_prob = aug_lidar_prob
        self.aug_both = aug_both
        self.filter_2d_gt = filter_2d_gt

        if skip_scenes is not None:
            import json
            skip_json = json.load(open(skip_scenes))
            skip_scenes = [item['scene_token'] for item in skip_json]
            self.skip_scenes = set(skip_scenes)
        else:
            self.skip_scenes = set()

    def set_iter(self, iter):
        self.iter = iter
        if not self.fixed_prob:
            self.set_prob()

    def set_prob(self):
        if self.iter < self.start_iter:
            self.prob = 0.0
        else:
            self.prob = self.st_prob * (self.iter - self.start_iter) / (self.max_iter - self.start_iter)

    def __call__(self, results):
        if self.skip_scenes and results.get('scene_token') in self.skip_scenes:
            return results

        if np.random.rand() > self.prob:
            return results
        
        if self.aug_both:
            augment_image = True
            augment_lidar = np.random.rand() < self.aug_lidar_prob
        else:
            augment_lidar = np.random.rand() < self.aug_lidar_prob
            augment_image = not augment_lidar
        
        imgs = results['img']
        h = imgs[0].shape[0]
        w = imgs[0].shape[1]
        self.d1 = 2
        self.d2 = min(h, w) * self.max_length_ratio
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)
        if self.ratio == 1:
            self.length = np.random.randint(1, d)
        else:
            self.length = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.length, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.length, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]

        mask = mask.astype(np.float32)
        mask = mask[:, :, None]
        if self.mode == 1:
            mask = 1 - mask
        elif self.mode == '01':
            if np.random.rand() > 0.5:
                mask = 1 - mask

        gridmask_info = dict(
            keep_masks=None,
            apply_image=bool(augment_image),
            apply_lidar=bool(augment_lidar),
            consistent=self.consistent
        )

        if augment_image or augment_lidar:
            keep_mask_single = mask.squeeze(-1).astype(np.uint8)
            gridmask_info['keep_masks'] = [keep_mask_single.copy() for _ in range(len(imgs))]

        results['gridmask_info'] = gridmask_info

        if augment_lidar:
            # point2image
            coords = results['points'][:, :3].tensor
            lidar2img = coords.new_tensor(np.asarray(results['lidar2img']))

            # lidar2img
            points_img = torch.cat([coords, torch.ones_like(coords[:, :1])], dim=1) @ lidar2img.permute(0, 2, 1)
            points_img[..., 2] = torch.clip(points_img[..., 2], min=1e-5, max=1e5)
            points_img[..., 0] /= points_img[..., 2]
            points_img[..., 1] /= points_img[..., 2]

            # mask the points that fall on mask of any image
            points = results['points']
            
            mask_tensor = points_img.new_tensor(mask)
            mask3d = torch.ones(points.shape[0], dtype=torch.bool)
            for i in range(len(imgs)):
                mask_value = torch.ones(points.shape[0], dtype=torch.bool)
                mask_indices = (points_img[i, :, 0] >= 0) & (points_img[i, :, 0] < w) & (points_img[i, :, 1] >= 0) & (points_img[i, :, 1] < h)
                mask_value[mask_indices] = mask_tensor[points_img[i, mask_indices, 1].long(), points_img[i, mask_indices, 0].long()].bool().squeeze()
                
                mask3d = mask3d & mask_value
            
            # If not in consistent mode, use the inverse mask for LiDAR.
            if not self.consistent:
                mask3d = ~mask3d

            points = points[mask3d]
            results.update(points=points)
            
        if augment_image:
            # Augment images.
            if self.offset:
                offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).float()
                offset = (1 - mask) * offset
                imgs = [x * mask + offset for x in imgs]
            else:
                imgs = [x * mask for x in imgs]
            
            results.update(img=imgs)
            
        if self.filter_2d_gt and augment_image and 'gridmask_info' in results and 'keep_masks' in results['gridmask_info']:
            # --- Filter heavily occluded 2D labels ---
            # Check and remove 2D box labels heavily occluded by GridMask (>90%).
            if 'gt_bboxes' in results and results['gt_bboxes'] is not None:
                mask_2d = mask.squeeze(-1)  # (H, W)
                new_gt_bboxes = []
                new_gt_labels = []
                new_centers2d = []
                new_depths = []
                
                # Iterate over each view.
                for view_idx in range(len(imgs)):
                    # Get 2D labels for the current view.
                    if len(results['gt_bboxes']) > view_idx:
                        view_gt_bboxes = results['gt_bboxes'][view_idx]
                        view_gt_labels = results.get('gt_labels', [None] * len(imgs))[view_idx] if 'gt_labels' in results else None
                        view_centers2d = results.get('centers2d', [None] * len(imgs))[view_idx] if 'centers2d' in results else None
                        view_depths = results.get('depths', [None] * len(imgs))[view_idx] if 'depths' in results else None
                        
                        # If the current view has 2D box labels.
                        if view_gt_bboxes is not None and len(view_gt_bboxes) > 0:
                            keep_indices = []
                            
                            # Check the occlusion level of each 2D box.
                            for box_idx, bbox in enumerate(view_gt_bboxes):
                                # bbox format: [x1, y1, x2, y2, ...]
                                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                
                                # Clip to the image bounds.
                                x1 = max(0, min(x1, w - 1))
                                y1 = max(0, min(y1, h - 1))
                                x2 = max(0, min(x2, w - 1))
                                y2 = max(0, min(y2, h - 1))
                                
                                # Skip invalid boxes.
                                if x2 <= x1 or y2 <= y1:
                                    continue
                                
                                # Compute the kept-pixel ratio inside the box; 1 in the mask means kept, 0 means occluded.
                                box_mask = mask_2d[y1:y2, x1:x2]
                                if box_mask.size > 0:
                                    visible_ratio = np.mean(box_mask)

                                    # Keep the 2D box if the visible ratio is >= 20% (occlusion < 80%).
                                    if visible_ratio >= 0.2:
                                        keep_indices.append(box_idx)
                            
                            # Filter 2D labels by kept indices.
                            if len(keep_indices) > 0:
                                keep_indices = np.array(keep_indices)
                                filtered_bboxes = view_gt_bboxes[keep_indices] if isinstance(view_gt_bboxes, np.ndarray) else [view_gt_bboxes[i] for i in keep_indices]
                                filtered_labels = view_gt_labels[keep_indices] if view_gt_labels is not None and isinstance(view_gt_labels, np.ndarray) else ([view_gt_labels[i] for i in keep_indices] if view_gt_labels is not None else None)
                                filtered_centers2d = view_centers2d[keep_indices] if view_centers2d is not None and isinstance(view_centers2d, np.ndarray) else ([view_centers2d[i] for i in keep_indices] if view_centers2d is not None else None)
                                filtered_depths = view_depths[keep_indices] if view_depths is not None and isinstance(view_depths, np.ndarray) else ([view_depths[i] for i in keep_indices] if view_depths is not None else None)
                            else:
                                # All boxes are heavily occluded; use empty arrays and keep dtype consistent.
                                bbox_dtype = view_gt_bboxes.dtype if isinstance(view_gt_bboxes, np.ndarray) else np.float32
                                filtered_bboxes = np.array([], dtype=bbox_dtype).reshape(0, 4) if isinstance(view_gt_bboxes, np.ndarray) else []
                                filtered_labels = np.array([], dtype=view_gt_labels.dtype if isinstance(view_gt_labels, np.ndarray) else np.int64) if view_gt_labels is not None and isinstance(view_gt_labels, np.ndarray) else ([] if view_gt_labels is not None else None)
                                filtered_centers2d = np.array([], dtype=view_centers2d.dtype if isinstance(view_centers2d, np.ndarray) else np.float32).reshape(0, 2) if view_centers2d is not None and isinstance(view_centers2d, np.ndarray) else ([] if view_centers2d is not None else None)
                                filtered_depths = np.array([], dtype=view_depths.dtype if isinstance(view_depths, np.ndarray) else np.float32) if view_depths is not None and isinstance(view_depths, np.ndarray) else ([] if view_depths is not None else None)
                            
                            new_gt_bboxes.append(filtered_bboxes)
                            if view_gt_labels is not None:
                                new_gt_labels.append(filtered_labels)
                            if view_centers2d is not None:
                                new_centers2d.append(filtered_centers2d)
                            if view_depths is not None:
                                new_depths.append(filtered_depths)
                        else:
                            # The current view has no 2D boxes; keep it unchanged and preserve dtype consistency.
                            new_gt_bboxes.append(view_gt_bboxes if view_gt_bboxes is not None else np.array([], dtype=np.float32).reshape(0, 4))
                            if view_gt_labels is not None or 'gt_labels' in results:
                                new_gt_labels.append(view_gt_labels if view_gt_labels is not None else np.array([], dtype=np.int64))
                            if view_centers2d is not None or 'centers2d' in results:
                                new_centers2d.append(view_centers2d if view_centers2d is not None else np.array([], dtype=np.float32).reshape(0, 2))
                            if view_depths is not None or 'depths' in results:
                                new_depths.append(view_depths if view_depths is not None else np.array([], dtype=np.float32))
                    else:
                        # Out of range; use empty arrays and keep dtype consistent.
                        new_gt_bboxes.append(np.array([], dtype=np.float32).reshape(0, 4))
                        if 'gt_labels' in results:
                            new_gt_labels.append(np.array([], dtype=np.int64))
                        if 'centers2d' in results:
                            new_centers2d.append(np.array([], dtype=np.float32).reshape(0, 2))
                        if 'depths' in results:
                            new_depths.append(np.array([], dtype=np.float32))
                
                # Update results.
                results['gt_bboxes'] = new_gt_bboxes
                if 'gt_labels' in results and len(new_gt_labels) > 0:
                    results['gt_labels'] = new_gt_labels
                if 'centers2d' in results and len(new_centers2d) > 0:
                    results['centers2d'] = new_centers2d
                if 'depths' in results and len(new_depths) > 0:
                    results['depths'] = new_depths
        
        # --- GT visibility computation ---
        gt_bboxes = results['gt_bboxes_3d']
        num_gts = len(gt_bboxes.tensor)
        gt_pts_visible_mask = gt_bboxes.tensor.new_ones(num_gts, dtype=torch.bool)
        gt_img_visible_mask = gt_bboxes.tensor.new_ones(num_gts, dtype=torch.bool)

        if augment_lidar:
            # LiDAR was augmented; check GT visibility in the point cloud.
            final_points = results['points'].tensor
            if num_gts > 0:
                if final_points.shape[0] > 0:
                    # Use NumPy ops on CPU for computation.
                    points_np = final_points.cpu().numpy()
                    boxes_np = gt_bboxes.tensor.cpu().numpy()
                    
                    # point_in_box_mask_np shape: (num_points, num_gts)
                    point_in_box_mask_np = points_in_rbbox(points_np, boxes_np[:, :7])

                    # points_per_box shape: (num_gts,)
                    points_per_box = point_in_box_mask_np.sum(axis=0)
                    gt_pts_visible_mask = torch.from_numpy(points_per_box > 0).to(gt_bboxes.tensor.device, dtype=torch.bool)
                else: # All points were filtered out.
                    gt_pts_visible_mask[:] = False
        
        if augment_image:
            # Images were augmented; check GT visibility in the images.
            if num_gts > 0:
                corners = gt_bboxes.corners # (num_gts, 8, 3)
                lidar2img = results.get('lidar2img')
                if lidar2img is None: # lidar2img may be missing.
                    return results
                if isinstance(lidar2img, list):
                    lidar2img = np.asarray(lidar2img)
                lidar2img = torch.from_numpy(lidar2img).float().to(corners.device)
                img_mask = torch.from_numpy(mask).squeeze(-1).to(corners.device) # (H, W)
                H, W = img_mask.shape

                for i in range(num_gts):
                    gt_is_visible_in_any_view = False
                    gt_corners = corners[i] # (8, 3)
                    gt_corners_hom = torch.cat([gt_corners, torch.ones_like(gt_corners[..., :1])], dim=-1) # (8, 4)
                    
                    points_img = lidar2img @ gt_corners_hom.T
                    points_img = points_img.permute(0, 2, 1) # (N_cam, 8, 4)

                    points_img[..., :2] /= points_img[..., 2:3].clamp(min=1e-5)

                    for cam_idx in range(points_img.shape[0]):
                        cam_points = points_img[cam_idx] # (8, 4)
                        depth = cam_points[:, 2]
                        points_xy = cam_points[:, :2]
                        
                        on_img = (points_xy[:, 0] >= 0) & (points_xy[:, 0] < W) & \
                                 (points_xy[:, 1] >= 0) & (points_xy[:, 1] < H) & \
                                 (depth > 1e-5)
                        
                        if not on_img.any():
                            continue

                        points_on_img = points_xy[on_img].long()
                        mask_values = img_mask[points_on_img[:, 1], points_on_img[:, 0]]
                        
                        if (mask_values == 1).sum() >= 2:
                            gt_is_visible_in_any_view = True
                            break
                    
                    if not gt_is_visible_in_any_view:
                        gt_img_visible_mask[i] = False
        
        results['gt_pts_visible_mask'] = gt_pts_visible_mask
        results['gt_img_visible_mask'] = gt_img_visible_mask

        return results

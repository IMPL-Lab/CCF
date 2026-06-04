# Copyright (c) MV2DFusion Project.
# Custom IoU calculator for BEV.

from __future__ import annotations

import torch

from mmdet.core.bbox import bbox_overlaps
from mmdet.core.bbox.iou_calculators.builder import IOU_CALCULATORS
from mmdet3d.core.bbox.structures import get_box_type


@IOU_CALCULATORS.register_module()
class BboxOverlapsBEV(object):
    """BEV IoU/GIoU Calculator (axis-aligned via nearest_bev).

    This calculator projects 3D boxes to BEV and computes 2D IoU/GIoU on
    axis-aligned BEV rectangles obtained by ``nearest_bev``. It mirrors the
    API style of :class:`BboxOverlaps3D` in mmdet3d.

    Note:
        - Only LiDAR coordinate is supported, as requested.
        - GIoU is computed on the axis-aligned BEV boxes (nearest_bev).
        - If ``is_aligned`` is ``False``, it computes pairwise overlaps.
          Otherwise it computes aligned overlaps.

    Args:
        coordinate (str): Coordinate system. Must be 'lidar'.
        mode (str): Default mode used when calling the calculator, one of
            {'iou', 'giou'}. Defaults to 'iou'.
    """

    def __init__(self, coordinate: str = 'lidar', mode: str = 'iou') -> None:
        assert coordinate in ['lidar'], (
            f"BboxOverlapsBEV currently only supports 'lidar', got {coordinate}")
        assert mode in ['iou', 'giou'], f'Unsupported mode {mode}'
        self.coordinate = coordinate
        self.mode = mode

    def __call__(
        self,
        bboxes1: torch.Tensor,
        bboxes2: torch.Tensor,
        mode: str | None = None,
        is_aligned: bool = False,
        eps: float | None = 1e-6,
    ) -> torch.Tensor:
        """Calculate BEV IoU/GIoU using axis-aligned BEV boxes.

        Args:
            bboxes1 (torch.Tensor): shape (N, 7+C), format
                (x, y, z, x_size, y_size, z_size, yaw, v*).
            bboxes2 (torch.Tensor): shape (M, 7+C).
            mode (str | None): 'iou' or 'giou'. If None, use `self.mode`.
            is_aligned (bool): Whether the calculation is aligned.
            eps (float | None): Epsilon for numerical stability when using
                `mode='giou'`. Forwarded to `bbox_overlaps`.

        Returns:
            torch.Tensor: If ``is_aligned`` is ``True``, return overlaps with
                shape (N,). If ``is_aligned`` is ``False``, return shape (N, M).
        """
        assert bboxes1.size(-1) == bboxes2.size(-1) and bboxes1.size(-1) >= 7
        _mode = self.mode if mode is None else mode
        assert _mode in ['iou', 'giou'], f'Unsupported mode {_mode}'

        box_type, _ = get_box_type(self.coordinate)
        boxes1 = box_type(bboxes1, box_dim=bboxes1.shape[-1])
        boxes2 = box_type(bboxes2, box_dim=bboxes2.shape[-1])

        bev1 = boxes1.nearest_bev  # (N, 4): x1,y1,x2,y2
        bev2 = boxes2.nearest_bev  # (M, 4)

        overlaps = bbox_overlaps(
            bev1, bev2, mode=_mode, is_aligned=is_aligned, eps=eps if eps is not None else 1e-6
        )
        return overlaps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(coordinate={self.coordinate}, mode={self.mode})"

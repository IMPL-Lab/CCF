import torch

from mmdet.core.bbox.samplers import PseudoSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.samplers.sampling_result import SamplingResult

@BBOX_SAMPLERS.register_module()
class IoUThresholdPseudoSampler(PseudoSampler):
    """A pseudo sampler with IoU threshold filtering for positive samples."""

    def __init__(self, iou_threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.iou_threshold = iou_threshold

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Sample positive and negative samples with IoU threshold filtering.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        # Get original positive and negative indices
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        
        # Apply IoU threshold filtering if max_overlaps is available
        if assign_result.max_overlaps is not None:
            # Get IoU values for positive samples
            pos_ious = assign_result.max_overlaps[pos_inds]
            
            # Filter positive samples based on IoU threshold
            iou_mask = pos_ious >= self.iou_threshold
            filtered_pos_inds = pos_inds[iou_mask]
            
            # Add filtered out positive samples to negative samples
            filtered_out_pos_inds = pos_inds[~iou_mask]
            # print(f"before: {neg_inds.shape}, after: {filtered_out_pos_inds.shape}")
            neg_inds = torch.cat([neg_inds, filtered_out_pos_inds])
            
            # Use filtered positive indices
            pos_inds = filtered_pos_inds
        
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
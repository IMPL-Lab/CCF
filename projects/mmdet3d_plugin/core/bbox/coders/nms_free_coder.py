import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox


@BBOX_CODERS.register_module()
class NMSFreeCoder(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 sigmoid=True):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.sigmoid = sigmoid

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds, pred_history=None, num_query_pts=0, modal_weights_img=None, modal_weights_pts=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
            modal_weights_img (Tensor, optional): Image modal weights, \
                shape [num_query, 1].
            modal_weights_pts (Tensor, optional): Point cloud modal weights, \
                shape [num_query, 1].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        raw_cls_scores = cls_scores.clone()
        if self.sigmoid:
            cls_scores = cls_scores.sigmoid()
        cls_scores = cls_scores.contiguous()
        max_num = min(cls_scores.view(-1).size(0), max_num)
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
        # Extract raw logits for all classes of the top-k queries
        raw_logits_for_topk_queries_all_classes = raw_cls_scores[bbox_index]
        bbox_preds = bbox_preds[bbox_index]

        
        if modal_weights_img is not None:
            modal_weights_img = modal_weights_img[bbox_index]
        
        if modal_weights_pts is not None:
            modal_weights_pts = modal_weights_pts[bbox_index]

        # Generate modal and query_id information
        # bbox_index is the original index of the selected query
        modal_info = []
        query_id_info = []
        for idx in bbox_index:
            if idx < num_query_pts:
                # Point-cloud query
                modal_info.append('pts')
                query_id_info.append(int(idx) + 1)  # query_id starts from 1
            else:
                # Image query
                modal_info.append('img')
                query_id_info.append(int(idx - num_query_pts) + 1)  # image query id starts from 1

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            logits = raw_logits_for_topk_queries_all_classes[mask]

            # Apply the same mask to modal and query_id information
            filtered_modal_info = [modal_info[i] for i in range(len(mask)) if mask[i]]
            filtered_query_id_info = [query_id_info[i] for i in range(len(mask)) if mask[i]]
            
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'logits': logits
            }
            
            # Add modal and query_id information
            predictions_dict['modal'] = filtered_modal_info
            predictions_dict['query_id'] = filtered_query_id_info
            
            # Add filtered prediction history to results
            if pred_history is not None:
                # pred_history['cls_scores']: [num_layers, num_queries, num_classes]
                # pred_history['bbox_preds']: [num_layers, num_queries, bbox_dim]
                # Apply the same bbox_index and mask to filter prediction history
                filtered_pred_history = {
                    'cls_scores': pred_history['cls_scores'][:, bbox_index][:, mask],  # [num_layers, num_filtered_queries, num_classes]
                    'bbox_preds': pred_history['bbox_preds'][:, bbox_index][:, mask]  # [num_layers, num_filtered_queries, bbox_dim]
                }
                predictions_dict['pred_history'] = filtered_pred_history

            if modal_weights_img is not None and modal_weights_pts is not None:
                predictions_dict['modal_weights_img'] = modal_weights_img[mask]
                predictions_dict['modal_weights_pts'] = modal_weights_pts[mask]
        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts_cls, preds_dicts_all=None):
        """Decode bboxes.
        Args:
            preds_dicts_cls (dict): Predictions from classification-related branches.
            preds_dicts_all (dict, optional): All predictions, including those not used for decoding.
        Returns:
            list[dict]: Decoded boxes.
        """
        if preds_dicts_all is None:
            preds_dicts_all = preds_dicts_cls
            
        all_cls_scores = preds_dicts_cls['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts_cls['all_bbox_preds'][-1]
        
        # Save full prediction history for output
        pred_history_cls = preds_dicts_cls['all_cls_scores']  # [num_layers, batch_size, num_queries, num_classes]
        pred_history_bbox = preds_dicts_cls['all_bbox_preds']  # [num_layers, batch_size, num_queries, bbox_dim]

        
        # Get num_query_pts to generate modal and query_id
        num_query_pts = preds_dicts_all.get('num_query_pts', 0)

        all_modal_weights_img = preds_dicts_all.get('all_modal_weights_img', None)
        all_modal_weights_pts = preds_dicts_all.get('all_modal_weights_pts', None)
        if all_modal_weights_img is not None:
            # average modal weights over all layers
            all_modal_weights_img = all_modal_weights_img.mean(dim=0)  # [batch_size, num_queries, 1]
        if all_modal_weights_pts is not None:
            # average modal weights over all layers
            all_modal_weights_pts = all_modal_weights_pts.mean(dim=0)  # [batch_size, num_queries, 1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            # Prepare prediction history for this batch
            batch_pred_history = {
                'cls_scores': pred_history_cls[:, i],  # [num_layers, num_queries, num_classes]
                'bbox_preds': pred_history_bbox[:, i]  # [num_layers, num_queries, bbox_dim]
            }
            
            # Pass all possible predictions to decode_single
            pred_dict = self.decode_single(
                all_cls_scores[i], 
                all_bbox_preds[i], 
                batch_pred_history,
                num_query_pts,
                all_modal_weights_img[i] if all_modal_weights_img is not None else None,
                all_modal_weights_pts[i] if all_modal_weights_pts is not None else None,
            )
            
            predictions_list.append(pred_dict)
        return predictions_list


@BBOX_CODERS.register_module()
class TrackingNMSFreeCoder(NMSFreeCoder):
    def __init__(self, tracking_decoding=False, **kwargs):
        super(TrackingNMSFreeCoder, self).__init__(**kwargs)
        self.tracking_decoding = tracking_decoding

    def decode_single(self, cls_scores, bbox_preds, obj_idxes=None, track_scores=None, ):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()

        if not self.tracking_decoding:
            max_num = min(cls_scores.view(-1).size(0), max_num)
            scores, indexs = cls_scores.view(-1).topk(max_num)
            labels = indexs % self.num_classes
            bbox_index = torch.div(indexs, self.num_classes, rounding_mode='floor')
            bbox_preds = bbox_preds[bbox_index]
            if obj_idxes is not None:
                obj_idxes = obj_idxes[bbox_index]
                track_scores = scores.clone()
        else:
            track_scores, indexs = cls_scores.max(dim=-1)
            labels = indexs % self.num_classes
            _, bbox_index = track_scores.topk(min(max_num, len(obj_idxes)))
            track_scores = track_scores[bbox_index]
            obj_idxes = obj_idxes[bbox_index]
            bbox_preds = bbox_preds[bbox_index]
            labels = labels[bbox_index]
            scores = track_scores

        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores >= self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)

            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]

            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
            }
            if obj_idxes is not None:
                track_scores = track_scores[mask]
                obj_idxes = obj_idxes[mask]
                predictions_dict['track_scores'] = track_scores
                predictions_dict['obj_idxes'] = obj_idxes

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        batch_size = all_cls_scores.size()[0]
        if 'obj_idxes' in preds_dicts.keys():
            obj_idxes = preds_dicts['obj_idxes'].clone()
            track_scores = [None for _ in range(batch_size)]
        else:
            obj_idxes = [None for _ in range(batch_size)]
            track_scores = [None for _ in range(batch_size)]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i], obj_idxes[i], track_scores[i], ))
        return predictions_list
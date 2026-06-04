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

import os.path as osp
import torch
import numpy as np
import mmcv
from mmdet3d.datasets import NuScenesDataset
from mmdet.datasets import DATASETS
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.api_wrappers import COCO
from mmcv.ops.nms import batched_nms
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox

@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, collect_keys, proposal_file=None, ann_2d_file=None, proposal_nms=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collect_keys = collect_keys

        self.proposal_file = proposal_file
        self.ann_2d_file = ann_2d_file
        self.proposal_nms = proposal_nms
        self.load_proposals()


    def load_proposals(self):
        if self.proposal_file is None:
            return

        if self.ann_2d_file is None:
            self.proposal_coco = COCO(self.proposal_file)
        else:
            class COCO_wrapper(COCO):
                def __init__(self, coco_file):
                    for k, v in vars(coco_file).items():
                        setattr(self, k, v)
                    self.img_ann_map = self.imgToAnns
                    self.cat_img_map = self.catToImgs

                def get_ann_ids(self, img_ids=[], cat_ids=[], area_rng=[], iscrowd=None):
                    return self.getAnnIds(img_ids, cat_ids, area_rng, iscrowd)

                def get_cat_ids(self, cat_names=[], sup_names=[], cat_ids=[]):
                    return self.getCatIds(cat_names, sup_names, cat_ids)

                def get_img_ids(self, img_ids=[], cat_ids=[]):
                    return self.getImgIds(img_ids, cat_ids)

                def load_anns(self, ids):
                    return self.loadAnns(ids)

                def load_cats(self, ids):
                    return self.loadCats(ids)

                def load_imgs(self, ids):
                    return self.loadImgs(ids)

            coco = COCO(self.ann_2d_file)
            proposal_coco = coco.loadRes(self.proposal_file)
            del coco
            self.proposal_coco = COCO_wrapper(proposal_coco)

        self.cat_ids = self.proposal_coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.impath_to_imgid = {}
        self.imgid_to_dataid = {}
        data_infos = []
        total_ann_ids = []
        for i in self.proposal_coco.get_img_ids():
            info = self.proposal_coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            self.impath_to_imgid['./data/nuscenes/' + info['file_name']] = i
            self.imgid_to_dataid[i] = len(data_infos)
            data_infos.append(info)
            ann_ids = self.proposal_coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{self.proposal_file}' are not unique!"
        self.proposal_infos = data_infos

    def impath_to_proposals(self, impath):
        img_id = self.impath_to_imgid[impath]
        data_id = self.imgid_to_dataid[img_id]
        ann_ids = self.proposal_coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.proposal_coco.load_anns(ann_ids)
        return self.get_proposals(self.proposal_infos[data_id], ann_info)

    def get_proposals(self, img_info, proposal_info):
        proposal_bboxes = []
        proposal_labels = []
        proposal_scores = []
        for i, ann in enumerate(proposal_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if not ann.get('iscrowd', False):
                proposal_bboxes.append(bbox)
                proposal_labels.append(self.cat2label[ann['category_id']])
                proposal_scores.append(ann['score'])

        if proposal_bboxes:
            proposal_bboxes = np.array(proposal_bboxes, dtype=np.float32)
            proposal_labels = np.array(proposal_labels, dtype=np.int64)
            proposal_scores = np.array(proposal_scores, dtype=np.float32)
        else:
            proposal_bboxes = np.zeros((0, 4), dtype=np.float32)
            proposal_labels = np.array([], dtype=np.int64)
            proposal_scores = np.array([], dtype=np.float32)

        if self.proposal_nms is not None and len(proposal_bboxes) > 0:
            score_thr = self.proposal_nms['score_thr']
            nms_cfg = self.proposal_nms.get('nms')
            class_agnostic = self.proposal_nms.get('class_agnostic', nms_cfg.get('class_agnostic'))
            max_num = self.proposal_nms.get('max_num', -1)

            # score filter
            valid_mask = proposal_scores > score_thr
            inds = valid_mask.nonzero()[0]
            proposal_bboxes, proposal_scores, proposal_labels = proposal_bboxes[inds], proposal_scores[inds], proposal_labels[inds]

            if len(proposal_bboxes) > 0:
                boxes = torch.from_numpy(proposal_bboxes).clone()
                scores = torch.from_numpy(proposal_scores).clone()
                nms_labels = labels = torch.from_numpy(proposal_labels).clone()
                if class_agnostic:
                    nms_labels = torch.zeros_like(labels)
                dets, keep = batched_nms(boxes, scores, nms_labels, nms_cfg)
                if max_num > 0:
                    dets = dets[:max_num]
                    keep = keep[:max_num]
                labels = labels[keep]

                proposal_bboxes = dets[:, :4].numpy()
                proposal_scores = dets[:, 4].numpy()
                proposal_labels = labels.numpy()

        proposals = np.concatenate([proposal_bboxes, proposal_scores[:, None], proposal_labels[:, None]], axis=-1, dtype=np.float32)
        return proposals

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.filter_empty_gt and \
            (example is None or ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return self.union2one([example])

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
        
    def union2one(self, queue):
        for key in self.collect_keys:
            if key != 'img_metas':
                queue[-1][key] = DC(torch.stack([each[key].data for each in queue]), cpu_only=False, stack=True, pad_dims=None)
            else:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
        for key in ['points', 'points_aug']:
            if key in queue[-1]:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)
        for key in ['proposals']:
            if key in queue[-1]:
                queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)
        if not self.test_mode:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
                if key == 'gt_bboxes_3d':
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=True)
                else:
                    queue[-1][key] = DC([each[key].data for each in queue], cpu_only=False)

        queue = queue[-1]
        return queue

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]

        # The timestamp is only used by LoadPointsFromMultiSweeps to compute
        # per-point time lag for aggregated LiDAR sweeps. It is not collected
        # into model metadata.
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)
                intrinsics.append(viewpad)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)
                
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                ))

            if self.proposal_file is not None:
                proposals = []
                for cam_i in range(len(image_paths)):
                    proposals_cam_i = self.impath_to_proposals(image_paths[cam_i])
                    proposals.append(proposals_cam_i)
                input_dict['proposals'] = proposals

        if not self.test_mode:
            annos = self.get_ann_info(index)
            annos.update( 
                dict(
                    bboxes=info['bboxes2d'],
                    labels=info['labels2d'],
                    bboxes3d_cams=info['bboxes3d_cams'],
                    centers2d=info['centers2d'],
                    depths=info['depths'],
                    bboxes_ignore=info['bboxes_ignore'])
            )
            input_dict['ann_info'] = annos
            
        return input_dict


    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            # 'v1.0-trainval': 'train',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)

        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det, self.with_velocity)
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            for i, box in enumerate(boxes):
                # import ipdb; ipdb.set_trace()
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)

                if 'logits' in det:
                    nusc_anno['logits'] = det['logits'][i].tolist()
                if 'pred_history' in det:
                    # pred_history['cls_scores']: [num_layers, num_queries, num_classes]
                    # pred_history['bbox_preds']: [num_layers, num_queries, bbox_dim]
                    nusc_anno['pred_history'] = {
                        'cls_scores': det['pred_history']['cls_scores'][:, i].tolist(),  # [num_layers, num_classes]
                        'bbox_preds': det['pred_history']['bbox_preds'][:, i].tolist()   # [num_layers, bbox_dim]
                    }
                if 'modal' in det:
                    nusc_anno['modal'] = det['modal'][i]  # String: 'pts' or 'img'
                if 'query_id' in det:
                    nusc_anno['query_id'] = det['query_id'][i]  # Integer: query index, starting from 1
                
                if 'modal_weights_img' in det and 'modal_weights_pts' in det:
                    # Save weights for both modalities
                    nusc_anno['modal_weights'] = {
                        'img': det['modal_weights_img'][i].tolist(),
                        'pts': det['modal_weights_pts'][i].tolist()
                    }

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix


def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list

def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
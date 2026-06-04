_base_ = [
    '../_base_/datasets/nus-3d.py',
    '../_base_/default_runtime.py',
]
plugin = True
plugin_dir = [
    'projects/mmdet3d_plugin/',
    'projects/isfusion/',
]

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# ISFusion setting
voxel_size = [0.075, 0.075, 0.2]
# point_cloud_range = [-54.4, -54.4, -5.0, 54.4, 54.4, 3.0]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]

res_factor = 1
out_size_factor = 8
voxel_shape = int((point_cloud_range[3]-point_cloud_range[0])//voxel_size[0])
bev_size = voxel_shape//out_size_factor
grid_size = [[bev_size, bev_size, 1], [bev_size//2, bev_size//2, 1]]
region_shape = [(6, 6, 1), (6, 6, 1)]
region_drop_info = [
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
]

# training hyperparameter
# base sample num: 16
num_gpus = 2
batch_size = 8
num_iters_per_epoch = 9050 // (num_gpus * batch_size)
num_epochs = 24

pts_ckpt = 'checkpoints/isfusion_source.pth'
img_ckpt = 'checkpoints/faster_rcnn_swint_fpn_source.pth'

roi_size = 7
roi_strides = [4, 8, 16, 32, 64]
model = dict(
    type='MV2DFusionV2Decouple',
    # eval_pts_query=True,
    dataset='nuscenes',
    position_level=2,
    use_grid_mask=True,

    loss_weight_3d=0.1,
    loss_weight_pts=1.,
    gt_mono_loss=True,
    use_decoupled_loss=False,

    img_backbone=dict(
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[0, 1, 2, 3],
        with_cp=True,
        convert_weights=False,
        frozen_stages=4, # all 4 layers
        # freeze=True,
        init_cfg=dict(type='Pretrained', checkpoint=img_ckpt, prefix='backbone.')
        ),
    img_neck=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            prefix='neck.', map_location='cpu'),
            type='FPN',
            in_channels=[96, 192, 384, 768],
            out_channels=256,
            num_outs=5),
    img_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', output_size=roi_size, sampling_ratio=-1),
        featmap_strides=roi_strides[:-1],
        out_channels=256, ),
    # faster rcnn
    img_roi_head=dict(
        type='TwoStageDetectorWrapper',
        init_cfg=dict(
            type='Pretrained', checkpoint=img_ckpt,
            map_location='cpu'),
        rpn_head=dict(
            type='RPNHead',
            in_channels=256,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                scales=[8],
                ratios=[0.5, 1.0, 2.0],
                strides=roi_strides),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[1.0, 1.0, 1.0, 1.0]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        ),
        roi_head=dict(
            type='StandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=roi_strides[:-1]),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=1.0),
                reg_decoded_bbox=True,
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)
            ),
        ),
        train_cfg=dict(
            rpn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            rpn_proposal=dict(
                nms_pre=2000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=True,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)),
        test_cfg=dict(
            min_bbox_size=4,
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.6, class_agnostic=True),
                max_per_img=60,)),
    ),
    img_query_generator=dict(
        type='ImageDistributionQueryGenerator',
        prob_bin=25,
        depth_range=[0.1, 90],
        gt_guided=False,
        gt_guided_loss=1.,
        with_cls=True,
        with_size=True,

        with_avg_pool=True,
        num_shared_convs=1,
        num_shared_fcs=1,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=roi_size,
        extra_encoding=dict(
            num_layers=2,
            feat_channels=[512, 256],
            features=[dict(type='intrinsic', in_channels=16, )]
        ),
    ),
    # isfusionv2
    pts_backbone=dict(
        init_cfg=dict(
            type='Pretrained', checkpoint=pts_ckpt, prefix='pts_backbone.', map_location='cpu'),
            type='ISFusionDetector',
            freeze=True,
            norm_eval=True,

            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,

            img_backbone=dict(
            type='SwinTransformer',
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            out_indices=[1, 2, 3],
            with_cp=False,
            convert_weights=False,
        ),
        img_neck=dict(
            type='GeneralizedLSSFPN',
            in_channels=[192, 384, 768],
            out_channels=256,
            start_level=0,
            num_outs=3),

        # pts
        pts_voxel_layer=dict(
            point_cloud_range=point_cloud_range,
            max_num_points=-1, voxel_size=voxel_size, max_voxels=(-1, -1)),
        pts_voxel_encoder=dict(
            type='DynamicVFEV2',
            in_channels=5 ,
            feat_channels=[64, 64],
            with_distance=False,
            voxel_size=voxel_size,
            with_cluster_center=True,
            with_voxel_center=True,
            point_cloud_range=point_cloud_range,
            norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
            # norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        ),
        pts_middle_encoder=dict(
            type='SparseEncoderV2',
            in_channels=64,
            sparse_shape=[41, voxel_shape, voxel_shape],
            base_channels=32,
            output_channels=256,
            order=('conv', 'norm', 'act'),
            encoder_channels=((32, 32, 64), (64, 64, 128), (128, 128, 256), (256, 256)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock',
        ),

        # multi-modal
        fusion_encoder=dict(
            type='ISFusionEncoderV2',
            num_points_in_pillar=20,
            embed_dims=256,
            bev_size=bev_size,
            num_views=6,
            region_shape=region_shape,
            grid_size=grid_size,
            region_drop_info=region_drop_info,
            instance_num=200,
        ),

        pts_backbone=dict(
            type='SECONDV2',
            in_channels=128,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),

        pts_neck=dict(
            type='SECONDFPNV2',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),

        pts_bbox_head = dict(
            type='TransFusionHeadV2',
            num_proposals=200,
            auxiliary=True,
            in_channels=256 * 2,
            hidden_channel=128,
            num_classes=len(class_names),
            num_decoder_layers=1,
            num_heads=8,
            nms_kernel_size=3,
            ffn_channel=256,
            dropout=0.1,
            bn_momentum=0.1,
            activation='relu',
            common_heads=dict(height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
            center_head = dict(center=[2, 2]),
            feat_modality = dict(use_lidar=True, use_camera=True),
            fusion_layer=dict(
                type='ConvFuser', in_channels=[128, 128], out_channels=128),
            center_decoder_cfg=dict(
                type='TransformerDecoderLayer',
                self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                norm_cfg=dict(type='LN'),
                pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)
            ),
            decoder_layer=dict(
                type='MFTransformerDecoderLayer',
                self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
                ffn_cfg=dict(
                    embed_dims=128,
                    feedforward_channels=256,
                    num_fcs=2,
                    ffn_drop=0.1,
                    act_cfg=dict(type='ReLU', inplace=True),
                ),
                norm_cfg=dict(type='LN'),
                pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)
            ),
            bbox_coder=dict(
                type='TransFusionBBoxCoderV2',
                pc_range=point_cloud_range[:2],
                voxel_size=voxel_size[:2],
                out_size_factor=out_size_factor,
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                score_threshold=0.0,
                code_size=10,
            ),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
            # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
            loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
            loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
            ),

        train_cfg=dict(
            pts=dict(
                dataset='nuScenes',
                assigner=dict(
                    type='HungarianAssigner3DV2',
                    iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                    cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                    reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                    iou_cost=dict(type='IoU3DCost', weight=0.25)
                ),
                pos_weight=-1,
                gaussian_overlap=0.1,
                min_radius=2,
                grid_size=[voxel_shape, voxel_shape, 40],  # [x_len, y_len, 1]
                voxel_size=voxel_size,
                out_size_factor=out_size_factor,
                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
                point_cloud_range=point_cloud_range)),
        test_cfg=dict(
            pts=dict(
                dataset='nuScenes',
                grid_size=[voxel_shape, voxel_shape, 40],
                out_size_factor=out_size_factor,
                pc_range=point_cloud_range[0:2],
                voxel_size=voxel_size[:2],
                nms_type=None,
                use_rotate_nms=True,  # only for TTA
                nms_thr=0.2,
                max_num=200,
            ))
    ),
    pts_query_generator=dict(
        type='PointCloudQueryGeneratorBEV',
        in_channels=128,
        hidden_channel=128,
        pts_use_cat=False,
    ),
    # fusion head
    fusion_bbox_head=dict(
        type='MV2DFusionHeadV2Decouple',
        sigma_reparam=True,
        prob_bin=25,
        post_bev_nms_ops=[0],
        num_classes=10,
        in_channels=256,
        num_query=300,
        topk_proposals=256,
        match_with_velo=True,
        scalar=10,  ##noise groups
        noise_scale=1.0,
        dn_weight=1.0,  ##dn loss weight
        split=0.75,  ###positive rate
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='MV2DFusionTransformerDecouple',
            decoder=dict(
                type='MV2DFusionTransformerDecoderDecouple',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='MV2DFusionTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MixedWeightedCrossAttention',
                            separate_weight=True,
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.,
                            attn_cfg=dict(
                                type='PETRMultiheadFlashAttention',
                                batch_first=False,
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,  ###use checkpoint to save memory
                    operation_order=(
                        'self_attn', 'norm',
                        'cross_attn', 'norm',
                        'ffn', 'norm')
                ),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0), ),
    train_cfg=dict(fusion=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3DV3',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=0.0),
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            pc_range=point_cloud_range,
            ), )))

# data pipeline
file_client_args = dict(backend='disk')
collect_keys = ['lidar2img', 'intrinsics', 'extrinsics']
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)  # we use nuimages pretrain for 2D detector
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (384, 1056)
ida_aug_conf = {
    "resize_lim": (0.57, 0.825),
    "final_dim": img_scale,
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}
ida_aug_conf_test = {
    "resize_lim": (0.72, 0.72),
    "final_dim": img_scale,
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": False,
    "training": False,
}


dataset_type = 'CustomNuScenesDataset'
data_root = './data/nuscenes/'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        # pad_empty_sweeps=True,
        # remove_close=True,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D',  with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=True),
    # dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf, training=False),
    dict(type='BEVGlobalRotScaleTrans',
         rot_range=[-1.57075, 1.57075],
         translation_std=[0, 0, 0],
         scale_ratio_range=[0.95, 1.05],
         reverse_angle=True,
         training=True,
         ),
    dict(type='BEVRandomFlip3D'),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='NormalizePoints'),
    dict(type='PointShuffle'),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PETRFormatBundle3D', class_names=class_names,
         collect_keys=collect_keys),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths'] + collect_keys,
         meta_keys=(
             'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
             'img_norm_cfg', 'gt_bboxes_3d', 'gt_labels_3d'), )
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        # pad_empty_sweeps=True,
        # remove_close=True,
        file_client_args=file_client_args),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeCropFlipRotImage', data_aug_conf=ida_aug_conf_test, training=False),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='NormalizePoints'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(img_scale[1], img_scale[0]),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'img'] + collect_keys ,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d',
                            'box_type_3d', 'img_norm_cfg', 'gt_bboxes_3d', 'gt_labels_3d'))
        ])
]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'splits/nuscenes_infos_train_singapore_norain_day_source.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        collect_keys=collect_keys + ['img', 'img_metas'],
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        ann_file=data_root + 'splits/nuscenes_infos_val_singapore_norain_day_source.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        collect_keys=collect_keys + ['img', 'img_metas'],
        ann_file=data_root + 'splits/nuscenes_infos_val_singapore_norain_day_source.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=4e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'pts_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

# optimizer_config = dict(
#     type='Fp16OptimizerHook', loss_scale='dynamic',
#     grad_clip=dict(max_norm=35, norm_type=2))

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))


# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ]
)

custom_hooks = []

evaluation = dict(interval=4 * num_iters_per_epoch, pipeline=test_pipeline)
checkpoint_config = dict(interval=1 * num_iters_per_epoch)
find_unused_parameters = True
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from = None
resume_from = None

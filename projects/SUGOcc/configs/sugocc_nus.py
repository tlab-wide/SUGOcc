_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(imports=['projects.SUGOcc.sugocc'], allow_failed_imports=False)
file_client_args = dict(backend='disk')
backend_args = None
sync_bn="torch"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
camera_used = ['left']

# 20 classes with unlabeled
class_names = [
    'others',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]
num_class = len(class_names)

point_cloud_range = [-40.0, -40.0, -1, 40.0, 40.0, 5.4]
occ_size = [200, 200, 16]
# downsample ratio in [x, y, z] when generating 3D volumes in LSS
lss_downsample = [1, 1, 1]
lss_occ_size = [occ_size[0] // lss_downsample[0],
                occ_size[1] // lss_downsample[1],
                occ_size[2] // lss_downsample[2]]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
voxel_size = [voxel_x, voxel_y, voxel_z]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

grid_config = {
    'xbound': [-40, 40, 0.4],
    'ybound': [-40, 40, 0.4],
    'zbound': [-1, 5.4, 0.4],
    'dbound': [1.0, 45.0, 0.5],
}

# settings for 3D encoder
numC_Trans = 80
voxel_channels = [numC_Trans, numC_Trans*2, numC_Trans*4, numC_Trans*8]
voxel_out_channels = 80
norm_cfg = dict(type='GN', num_groups=20, requires_grad=True)

# settings for mask2former head
mask2former_num_queries = 108
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3 # divided by ndim
mask2former_num_heads = voxel_out_channels // 32

model = dict(
    type='SUGOcc',
    img_backbone=dict(
        type='MMDetWrapper',
        config_path='maskdino/configs/maskdino_r50_8xb2-panoptic-export.py',
        custom_imports='maskdino',
        num_outs=4,
        checkpoint_path='ckpts/maskdino_r50_50e_300q_panoptic_pq53.0.pth'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 256, 256, 256, 256],
        upsample_strides=[0.25, 0.5, 1,2,4],
        out_channels=[128, 128, 128, 128, 128]),
    img_view_transformer=dict(
        type='ViewTransformerLiftSplatShootVoxel',
        numC_input=640,
        cam_channels=27,
        loss_depth_weight=1.0,
        downsample=16,
        num_classes=num_class,
        grid_config=grid_config,
        data_config=data_config,
        numC_Trans=numC_Trans,
        empty_idx=17,
        vp_megvii=False,
        lss_downsample=lss_downsample,
        grid_size=lss_occ_size,
    ),
    img_bev_encoder_backbone=dict(
        type='MinkowskiUNetEncoder',
        in_channels=voxel_channels,
        num_blocks=[2,2,2,2],
        num_dense_blocks=3,
        kernel_size=3,
        cross_kernel=True,
        voxel_size=lss_occ_size,
        out_channels=[voxel_out_channels for _ in voxel_channels],
    ),
    img_bev_encoder_neck=dict(
        type='SparseGenerativePixelDecoder',
        nclasses=num_class,
        dataset='nuscenes',
        empty_idx=17,
        in_channels=voxel_channels,
        out_channels=voxel_channels,
        process_block_num=2,
        process_kernel_size=3,
        process_cross_kernel=True,
        soft_pruning=True,
        pruning_ratio=[0.1, 0.1, 0.1],
        is_pruning=[True, True, True],
        voxel_size=lss_occ_size,
        lss_downsample=lss_downsample,
    ),
    pts_bbox_head=dict(
        type='SparseOCRMask2OccHead',
        feat_channels=mask2former_feat_channel,
        out_channels=mask2former_output_channel,
        lss_downsample=lss_downsample,
        norm_cfg=norm_cfg,
        dataset='nuscenes',
        num_queries=mask2former_num_queries,
        num_occupancy_classes=num_class,
        pooling_attn_mask=True,
        sparse_input=True,
        sample_weight_gamma=0.25,
        cascade_ratio=2,
        dn_num=20,
        fine_topk=30000,
        empty_idx=17,
        final_occ_size=occ_size,
        sample_from_voxel=True,
        sample_from_img=False,
        grid_config=grid_config,
        data_config=data_config,
        # using stand-alone pixel decoder
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=mask2former_pos_channel, normalize=True),
        # using the original transformer decoder
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=1,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=mask2former_feat_channel,
                    num_heads=mask2former_num_heads,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=mask2former_feat_channel,
                    feedforward_channels=mask2former_feat_channel * 8,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=mask2former_feat_channel * 8,
                operation_order=('cross_attn', 'norm','self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        # loss settings
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_class + [0.1]),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        point_cloud_range=point_cloud_range,
    ),
    train_cfg=dict(
        pts=dict(
            num_points=12544 * 4,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='MaskHungarianAssigner',
                cls_cost=dict(type='ClassificationCost', weight=2.0),
                mask_cost=dict(
                    type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                dice_cost=dict(
                    type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
            sampler=dict(type='MaskPseudoSampler'),
        )),
    test_cfg=dict(
        pts=dict(
            semantic_on=True,
            panoptic_on=False,
            instance_on=False)),
)

dataset_type = 'CustomNuScenesDatasetOccupancy'
data_root = 'data/nuscenes/'
ann_file = 'labels'

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=True),
    dict(type='LoadOccGTFromFile',ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config),
    dict(type='OccDefaultFormatBundle3D'),
    dict(
        type='CustomPack3DDetInputs', keys=['img_inputs', 'gt_occ',],
        meta_keys=['pc_range', 'occ_size','sample_idx','filename', ]),
]

test_pipeline = [
    dict(type='PrepareImageInputs', data_config=data_config, sequential=False),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        is_train=False),
    dict(type='LoadOccGTFromFile',ignore_nonvisible=True),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='OccDefaultFormatBundle3D'),
            dict(type='CustomPack3DDetInputs', keys=['img_inputs', 'gt_occ'],
                 meta_keys=['pc_range', 'occ_size','sample_idx','filename']),
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        img_info_prototype='bevdet',
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='bevdetv2-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        img_info_prototype='bevdet',
        occ_size=occ_size,
        pc_range=point_cloud_range,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(type='CustomOccMetricNuscenes')

test_evaluator = val_evaluator
# learning policy
param_scheduler = dict(type='MultiStepLR', by_epoch=True, milestones=[20, 25], gamma=0.1)

# for most of these optimizer settings, we follow Mask2Former
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=2e-4,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999),),
    paramwise_cfg=dict(
            custom_keys={
                'query_embed': embed_multi,
                'query_feat': embed_multi,
                'level_embed': embed_multi,
                'absolute_pos_embed': dict(decay_mult=0.),
                'relative_position_bias_table': dict(decay_mult=0.),
                'img_backbone': dict(lr_mult=0.1, decay_mult=1.0),
            },
            norm_decay_mult=0.0),
    clip_grad=dict(max_norm=5, norm_type=2))

# runtime settings
randomness=dict(seed=0)
train_cfg = dict(by_epoch=True, max_epochs=24, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=2)
log_processor = dict(window_size=50)
# trace_config = dict(type='log_trace')
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', interval=1),
)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]
find_unused_parameters=True
_base_ = [
    "_base_/default_runtime.py",
]

work_dir = "work_dirs/pointmapnet_large_range"
custom_imports = dict(
    imports=[
        "plugin",
    ],
    allow_failed_imports=False,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
patch_size = (120.0, 60.0)
map_classes = ["divider", "ped_crossing", "boundary"]
num_class = len(map_classes)
num_points = 30
pc_range = [-30.0, -60.0, -3.0, 30.0, 60.0, 3.0]
_dim_ = 256
num_level = 1
use_norm_coord = True

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
model = dict(
    type="PMNTR",
    use_grid_mask=True,
    data_preprocessor=dict(type="MapDataPreprocessor"),
    img_backbone=dict(
        type="ResNet",
        _scope_="mmdet",
        depth=50,
        out_indices=(
            3,
        ),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=False),
        norm_eval=True,
        style="pytorch",
        init_cfg=dict(type="Pretrained", checkpoint="ckpts/resnet50-19c8e357.pth"),
    ),
    img_neck=dict(
        type="FPN",
        _scope_="mmdet",
        in_channels=[2048],
        out_channels=_dim_,
        num_outs=num_level,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ),
    head=dict(
        type="PMNHead",
        num_classes=num_class,
        num_points=num_points,
        coord_dim=2,
        num_raw_point=2000,
        with_box_refine=True,
        transformer=dict(
            type="PMNTransformer",
            num_feature_levels=num_level,
            pc_range=pc_range,
            num_points=num_points,
            positional_encoding=dict(
                type='SinePositionalEncoding3D', num_feats=128, normalize=True),
            encoder=dict(
                type="PointEncoderV1",
                return_intermediate=True,
                num_layers=3,
                pc_range=pc_range,
                zs=[0.2, 0.3, 0.4, 0.5],
                transformerlayers=dict(
                    type="LineLayer",
                    embed_dims=_dim_,
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                            batch_first=True,
                        ),
                        dict(
                            type="SpatialCrossAttentionV1",
                            embed_dims=_dim_,
                            deformable_attention=dict(
                                embed_dims=_dim_,
                                num_levels=num_level,
                                num_points=8,
                                im2col_step=192,
                                type='MSDeformableAttention3D'),
                        ),
                    ],
                    feedforward_channels=_dim_ * 2,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
            decoder=dict(
                type="PMNDecoder",
                num_layers=3,
                num_points=num_points,
                distance=0.1,
            ),
        ),
        bbox_coder=dict(
            type="LineTRNMSFreeCoder",
            pc_range=pc_range,
            max_num=50,
            num_classes=num_class,
        ),
        line_assigner=dict(
            type="LineAssigner",
            cls_cost=dict(type="FocalLossCost", _scope_="mmdet", weight=2.0),
            pts_cost=dict(type="OrderedPtsL1Cost", weight=5.),
            pc_range=pc_range,
        ),
        point_assigner=dict(
            type="PointAssigner",
            cls_cost=dict(type="FocalLossCost", _scope_="mmdet", weight=20.0),
            pts_cost=dict(type="PointL1Cost", weight=200.0),
            pc_range=pc_range,
        ),
        loss_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.,
        ),
        loss_pts=dict(type="PtsL1Loss", loss_weight=5.),
        loss_point_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=80.0,
        ),
        loss_point_mask=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=40.0,
        ),
        loss_point_pts=dict(type="PtsL1Loss", loss_weight=800.0),
    ),
)

backend_args = None
data_root = "data/nuscenes"

train_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     backend_args=backend_args),
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="VectorizeLocalMapV1",
        data_root=data_root,
        patch_size=patch_size,
        map_classes=map_classes,
        fixed_ptsnum_per_line=num_points,
        cache_dir=data_root + "/temp1",
        use_cache=False,
        use_focus_point=False,
    ),
    dict(type="AddRawPoint"),
    dict(type="PositionEmbeddings", pc_range=pc_range, feat_shapes=[[15, 25]]),
    dict(
        type="PackMapInputs",
        keys=["img", "gt_labels_3d", "gt_bboxes_3d", "raw_points", "raw_points_label", "point_masks"],
        meta_keys=("sample_idx", "pad_shape", "img_shape", "ori_shape", "lidar2img", "ego2global_translation",
                   "ego2global_rotation", "pos_points"),
    ),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="NuscDataset",
        data_root=data_root,
        ann_file="nuscenes_maptr_infos_train.pkl",
        modality=input_modality,
        pipeline=train_pipeline,
    ),
    pin_memory=True,
)

test_pipeline = [
    # dict(
    #     type='LoadPointsFromFile',
    #     coord_type='LIDAR',
    #     load_dim=5,
    #     use_dim=5,
    #     backend_args=backend_args),
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.5]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="VectorizeLocalMapV1",
        data_root=data_root,
        patch_size=patch_size,
        map_classes=map_classes,
        fixed_ptsnum_per_line=num_points,
        cache_dir=data_root + "/temp1",
        use_cache=False,
        use_focus_point=False,
    ),
    dict(type="PositionEmbeddings", pc_range=pc_range, feat_shapes=[[15, 25]]),
    dict(
        type="PackMapInputs",
        keys=["img", "gt_labels_3d", "gt_bboxes_3d", "raw_points", "raw_points_label"],
        meta_keys=("sample_idx", "pad_shape", "img_shape", "ori_shape", "lidar2img", "ego2global_translation",
                   "ego2global_rotation", "img_filenames", "pos_points"),
    ),
]

test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="NuscDataset",
        data_root=data_root,
        ann_file="nuscenes_maptr_infos_val.pkl",
        modality=input_modality,
        pipeline=test_pipeline,
    ),
    pin_memory=True,
)

test_evaluator = dict(
    type="VectorMapMetricV1",
    data_root=data_root,
    ann_file="nuscenes_maptr_infos_val.pkl",
    pc_range=pc_range,
    fixed_num=num_points,
    eval_cfg=dict(
        num_class=3,
        class_name=["ped_crossing", "divider", "contours"],
        coord_dim=2,
    ),
)

val_cfg = None
lr = 6e-4
max_epochs = 24
optim_wrapper = dict(
    optimizer=dict(lr=lr),
    type="AmpOptimWrapper",
    loss_scale="dynamic",
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type="CosineAnnealingLR",
        begin=0,
        by_epoch=True,
        eta_min_ratio=1e-3,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)

auto_scale_lr = dict(enable=True, base_batch_size=32)
# find_unused_parameters = True
# load_from = "work_dirs/linetrv1/epoch_4.pth"
# resume = True
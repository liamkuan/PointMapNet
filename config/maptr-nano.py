_base_ = [
    "_base_/default_runtime.py",
]

work_dir = "out"
custom_imports = dict(
    imports=[
        "plugin",
    ],
    allow_failed_imports=False,
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
patch_size = (60.0, 30.0)
map_classes = ["divider", "ped_crossing", "boundary"]
num_class = len(map_classes)
num_points = 20
pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 80
bev_w_ = 40
queue_length = 1

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=True
)
model = dict(
    type="MapTR",
    use_grid_mask=True,
    data_preprocessor=dict(
        type="MapDataPreprocessor",
    ),
    img_backbone=dict(
        type="ResNet",
        _scope_="mmdet",
        depth=18,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        init_cfg=dict(type="Pretrained", checkpoint="ckpts/resnet18-f37072fd.pth"),
    ),
    img_neck=dict(
        type="FPN",
        _scope_="mmdet",
        in_channels=[512],
        out_channels=_dim_,
        start_level=0,
        num_outs=_num_levels_,
        add_extra_convs="on_output",
        relu_before_extra_convs=True,
    ),
    head=dict(
        type="MapTRHead",
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=900,
        num_vec=100,
        num_pts_per_vec=num_points,  # one bbox
        num_pts_per_gt_vec=num_points,
        dir_interval=1,
        query_embed_type="instance_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v2",
        num_classes=num_class,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type="MapTRPerceptionTransformer",
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            # num_feature_levels=_num_levels_,
            encoder=dict(
                type="BEVFormerEncoder",
                num_layers=1,
                pc_range=pc_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type="BEVFormerLayer",
                    attn_cfgs=[
                        dict(
                            type="TemporalSelfAttention", embed_dims=_dim_, num_levels=1
                        ),
                        dict(
                            type="GeometrySptialCrossAttention",
                            pc_range=pc_range,
                            attention=dict(
                                type="GeometryKernelAttention",
                                embed_dims=_dim_,
                                num_heads=4,
                                dilation=1,
                                kernel_size=(3, 5),
                                num_levels=1,
                                im2col_step=192
                            ),
                            embed_dims=_dim_,
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
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
                type="MapTRDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="CustomMSDeformableAttention",
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=4,
                            im2col_step=192
                        ),
                    ],
                    ffn_cfgs=dict(
                        type="FFN",
                        embed_dims=_dim_,
                        feedforward_channels=_ffn_dim_,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type="ReLU", inplace=True),
                    ),
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
        ),
        bbox_coder=dict(
            type="MapTRNMSFreeCoder",
            # post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            post_center_range=[-20, -35, -20, -35, 20, 35, 20, 35],
            pc_range=pc_range,
            max_num=50,
            voxel_size=[0.15, 0.15, 4],
            num_classes=num_class,
        ),
        assigner=dict(
            type="LineAssigner",
            cls_cost=dict(type="FocalLossCost", _scope_="mmdet", weight=2.0),
            pts_cost=dict(type="OrderedPtsL1Cost", weight=5.0),
            pc_range=pc_range,
        ),
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            _scope_="mmdet",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        loss_cls=dict(
            type="FocalLoss",
            _scope_="mmdet",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        ),
        loss_pts=dict(type="PtsL1Loss", loss_weight=5.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
    ),
)

backend_args = None
data_root = "data/nuscenes"

train_pipeline = [
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.2]),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="VectorizeLocalMapV1",
        data_root=data_root,
        patch_size=patch_size,
        map_classes=map_classes,
        fixed_ptsnum_per_line=num_points,
        cache_dir=data_root + "/temp1",
        use_cache=True,
    ),
    dict(
        type="PackMapInputs",
        keys=["img", "gt_labels_3d", "gt_bboxes_3d"],
        meta_keys=(
            "sample_idx",
            "pad_shape",
            "img_shape",
            "ori_shape",
            "lidar2img",
            "can_bus",
            "ego2global_translation",
            "ego2global_rotation",
        ),
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImagesFromFiles", to_float32=True),
    dict(type="RandomScaleImageMultiViewImage", scales=[0.2]),
    dict(type="Normalize3D", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="VectorizeLocalMapV1",
        data_root=data_root,
        patch_size=patch_size,
        map_classes=map_classes,
        fixed_ptsnum_per_line=num_points,
        cache_dir=data_root + "/temp1",
        use_cache=True,
    ),
    dict(
        type="PackMapInputs",
        keys=["img", "gt_labels_3d", "gt_bboxes_3d"],
        meta_keys=(
            "sample_idx",
            "pad_shape",
            "img_shape",
            "ori_shape",
            "lidar2img",
            "can_bus",
            "ego2global_translation",
            "ego2global_rotation",
        ),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
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

test_dataloader = dict(
    batch_size=1,
    num_workers=0,
    sampler=dict(type="DefaultSampler", shuffle=False),
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
    loss_scale=512.0,
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

# find_unused_parameters=True
# load_from = "out/epoch_24.pth"
# resume = True

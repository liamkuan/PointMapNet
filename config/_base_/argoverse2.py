input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
class2label = {
    'ped_crossing': 0,
    'divider': 1,
    'others': -1,
    # 'centerline': 3, 1
}
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
pt_range = [-32, -64, -3, 32, 64, 0]  # x_min, y_min, z_min x_max, y_max, z_max 激光雷达坐标系
num_points_per_meter = 2

train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeMultiViewImages',
         size=(384, 512),  # H, W
         change_intrinsics=True,),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=False),
    dict(type='AV2LoadVectorMap',
         map_range=[-60, -30, 60, 30],
         class2label=class2label),
    # dict(type='AV2LoadRasterizeDrivableArea', densities=2),
    dict(type='PaddingTarget', num_point=20, dim=2),
    # dict(type='GenerateReferencePoints',
    #      pt_range=pt_range,
    #      num_points_in_pillar=4,
    #      num_points_per_meter=num_points_per_meter),
    dict(type='PackMapInputs',
        #  keys=['img', 'ref_points_cam', 'bev_mask', 'bev_points'],
         keys=['img', ],
        #  meta_keys=('target', 'img_shape', 'ori_shape', 'pad_shape', 'ego2img', 'cam_timestamp', 'drivable_area')),
         meta_keys=('target', 'img_shape', 'ori_shape', 'pad_shape', 'ego2img')),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    # batch_size=1,
    # num_workers=2,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    dataset=dict(type='AV2Dataset',
                 data_root='./data/av2',
                 ann_file='argoverse2_map_infos_train.pkl',
                 modality=input_modality,
                 cat2id=class2label,
                 pipeline=train_pipeline,
                 prev_frame=True,),
    pin_memory=True)

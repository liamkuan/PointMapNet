data_root = './data/nuscenes'
pt_range = [-20, -40, -3, 20, 40, 0]  # x_min, y_min, z_min x_max, y_max, z_max 激光雷达坐标系
num_points_per_meter = 2
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
roi_size = (60, 30)
canvas_size = (200, 100)
thickness = 3
num_points = 30
class2label = {
    'ped_crossing': 0,
    'divider': 1,
    'contours': 2,
    'others': -1,
    # 'centerline': 3,
}
# vectorize params
coords_dim = 2
sample_dist = 1.0
sample_num = -1

backend_args = None

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

num_class = max(list(class2label.values())) + 1


train_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles'),
    dict(type='ResizeMultiViewImages',
         size=(int(128 * 2), int((16 / 9 * 128) * 2)),  # H, W
         change_intrinsics=True,
         ),
    dict(
        type='VectorizeLocalMap',
        data_root=data_root,
        patch_size=(roi_size[1], roi_size[0]),
        sample_dist=0.7,
        num_samples=150,
        sample_pts=False,
        max_len=num_points,
        padding=False,
        normalize=True,
        fixed_num={
            'ped_crossing': -1,
            'divider': -1,
            'contours': -1,
            'others': -1,
        },
        class2label=class2label,
        centerline=False,
    ),
    dict(type='Vectors2Target'),
    # dict(
    #     type='PolygonizeLocalMapBbox',
    #     canvas_size=canvas_size,  # xy
    #     coord_dim=2,
    #     num_class=num_class,
    #     mode='xyxy',
    #     test_mode=True,
    #     threshold=4 / 200,
    #     flatten=False,
    # ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32, change_intrinsics=True),
    # dict(type='LoadPointsFromFile',
    #      coord_type='LIDAR',
    #      load_dim=5,
    #      use_dim=3,
    #      ),
    # dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='Points2ReferencePoints', num_points=num_query),
    dict(type='GenerateReferencePoints',
         pt_range=pt_range,
         num_points_in_pillar=4,
         num_points_per_meter=num_points_per_meter),
    dict(type='PackMapInputs',
         keys=['img', 'ref_points_cam', 'bev_mask', 'bev_points'],
         meta_keys=('sample_idx', 'target', 'vectors')),
]


eval_cfg = dict(
    patch_size=roi_size,
    origin=(-30, -15),
    evaluation_cfg=dict(
        result_path='./',
        dataroot=data_root,
        # will be overwirte in code
        ann_file='nuscenes_map_infos_val.pkl',
        num_class=num_class,
        class_name=['ped_crossing', 'divider', 'contours'],
    )
)

train_dataloader = dict(
    # batch_size=2,
    # num_workers=8,
    batch_size=1,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    # collate_fn=dict(type='pseudo_collate'),
    dataset=dict(type='NuscDataset',
                 data_root='./data/nuscenes',
                 ann_file='nuscenes_map_infos_train.pkl',
                 modality=input_modality,
                 roi_size=roi_size,
                 cat2id=class2label,
                 pipeline=train_pipeline,
                 interval=1),
    pin_memory=True)
# val_dataloader = dict(
#     batch_size=2,
#     num_workers=8,
#     sampler=dict(
#         type='DefaultSampler',
#         shuffer=True,
#     ),
#     dataset=dict(
#         type='NuscDataset',
#         data_root='./data/nuscenes',
#         ann_file='nuscenes_map_infos_val.pkl',
#         modality=input_modality,
#         roi_size=roi_size,
#         cat2id=class2label,
#         pipeline=train_pipeline,
#         interval=1),
#     pin_memory=True)
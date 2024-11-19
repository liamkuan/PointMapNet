default_scope = 'mmdet3d'

work_dir = None
lr = 2e-4
max_epochs = 80
optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

param_scheduler = [
    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        # T_max=max_epochs,
        # end=max_epochs,
        by_epoch=True,
        eta_min_ratio=1e-3,)
]
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=True, base_batch_size=8)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=4),
    dist_cfg=dict(backend='nccl'),
)

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
visualizer = dict(type='Visualizer', vis_backends=[dict(type='TensorboardVisBackend')])
log_level = 'INFO'
load_from = None
resume = False

# configs/unet/unet_mobilenetv3_agronav_512x512.py

norm_cfg = dict(type='BN', requires_grad=True)
crop_size = (512, 512)
num_classes = 9

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        size=crop_size,
        mean=[123.675, 116.28, 103.53],  # ImageNet mean
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='MobileNetV3',
        arch='small',
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://mobilenet_v3_small')
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=576,  # change according to backbone output
        channels=128,
        num_convs=1,
        kernel_size=1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

dataset_type = 'CustomDataset'
data_root = 'data/agronav_dataset'

classes = ['soil', 'crop', 'weed', 'sky', 'human', 'vehicle', 'building', 'fence', 'other']
palette = [
    [128, 64, 128],   # soil
    [244, 35, 232],   # crop
    [70, 70, 70],     # weed
    [135, 206, 235],  # sky
    [220, 20, 60],    # human
    [0, 0, 142],      # vehicle
    [70, 130, 180],   # building
    [100, 40, 40],    # fence
    [200, 200, 200]   # other
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/train',
            seg_map_path='ann_dir/train'
        ),
        pipeline=train_pipeline,
        classes=classes,
        palette=palette
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val',
            seg_map_path='ann_dir/val'
        ),
        pipeline=test_pipeline,
        classes=classes,
        palette=palette
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

# Schedule
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=1000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=None
)

# Logging
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=100),
    checkpoint=dict(type='CheckpointHook', interval=1000),
    timer=dict(type='IterTimerHook')
)

env_cfg = dict(cudnn_benchmark=True)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_level = 'INFO'
load_from = None
resume = False
work_dir = './work_dirs/unet_mobilenetv3_agronav'

# Set fixed seed for reproducibility
randomness = dict(seed=0)

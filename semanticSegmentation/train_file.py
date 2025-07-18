from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

# === Dataset meta ===
data_root = 'data/dataset'
train_img_dir = 'img_dir/train'
train_ann_dir = 'ann_dir/train'
val_img_dir = 'img_dir/val'
val_ann_dir = 'ann_dir/val'

classes = ('soil', 'crop', 'weed', 'sky', 'human', 'vehicle', 'building', 'fence', 'other')
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

# === Register dataset ===
@DATASETS.register_module()
class AgronavDataset(BaseSegDataset):
    METAINFO = dict(classes=classes, palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

def main():
    from mmengine import Config
    from mmengine.runner import Runner

    # === Load base config for UNet + FCN with ResNet50 backbone ===
    cfg = Config.fromfile('configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py')

    # === Basic config updates ===
    cfg.crop_size = (512, 512)
    cfg.model.data_preprocessor.size = cfg.crop_size
    cfg.norm_cfg = dict(type='BN', requires_grad=True)

    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.num_classes = 9
    cfg.model.auxiliary_head = None

    cfg.dataset_type = 'AgronavDataset'
    cfg.data_root = data_root

    # === Simplified Data pipeline without complex augmentations ===
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', scale=cfg.crop_size, keep_ratio=True),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PackSegInputs')
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=cfg.crop_size, keep_ratio=True),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs')
    ]

    cfg.train_dataloader = dict(
        batch_size=4,
        num_workers=2,
        persistent_workers=True,
        sampler=dict(type='InfiniteSampler', shuffle=True),
        dataset=dict(
            type=cfg.dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path=train_img_dir, seg_map_path=train_ann_dir),
            pipeline=cfg.train_pipeline
        )
    )

    cfg.val_dataloader = dict(
        batch_size=1,
        num_workers=2,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=cfg.dataset_type,
            data_root=data_root,
            data_prefix=dict(img_path=val_img_dir, seg_map_path=val_ann_dir),
            pipeline=cfg.test_pipeline
        )
    )

    cfg.test_dataloader = cfg.val_dataloader

    cfg.val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
    cfg.test_evaluator = cfg.val_evaluator

    cfg.train_cfg = dict(type='IterBasedTrainLoop', max_iters=200, val_interval=20)
    cfg.val_cfg = dict(type='ValLoop')
    cfg.test_cfg = dict(type='TestLoop')

    cfg.optim_wrapper = dict(
        type='OptimWrapper',
        optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005),
        clip_grad=None
    )

    cfg.param_scheduler = [
        dict(
            type='PolyLR',
            eta_min=1e-5,
            power=0.9,
            begin=0,
            end=200,
            by_epoch=False
        )
    ]

    cfg.default_hooks.logger = dict(type='LoggerHook', interval=20)
    cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=20)
    cfg.default_hooks.timer = dict(type='IterTimerHook')
    cfg.default_hooks.pop('visualization', None)  # Disabled visualization safely

    cfg.visualizer = None

    cfg.env_cfg = dict(cudnn_benchmark=True)
    cfg.log_level = 'INFO'
    cfg.load_from = None
    cfg.resume = False
    cfg.work_dir = './work_dirs/unet_fcn_resnet50_agronav'
    cfg.randomness = dict(seed=0)

    from mmengine import mkdir_or_exist
    mkdir_or_exist(cfg.work_dir)

    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.train()

if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()
    main()

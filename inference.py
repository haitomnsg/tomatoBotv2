import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from mmseg.visualization import SegLocalVisualizer

# === Paths ===
CONFIG_FILE = 'semanticSegmentation/mmsegmentation/configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py'
CHECKPOINT_FILE = 'semanticSegmentation/mmsegmentation/work_dirs/unet_fcn_resnet50_agronav/epoch_1.pth'
IMAGE_PATH = 'semanticSegmentation/mmsegmentation/demo/agronav_test.jpg'
OUTPUT_PATH = 'semanticSegmentation/mmsegmentation/demo/agronav_test.png'

# === Dataset METAINFO ===
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

# === Load config and modify metainfo ===
cfg = Config.fromfile(CONFIG_FILE)
cfg.model.data_preprocessor.update({'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'bgr_to_rgb': True})
cfg.model.test_cfg = dict(mode='whole')
cfg.model.decode_head.num_classes = len(classes)
cfg.model.data_preprocessor.size = (512, 512)
cfg.model.data_preprocessor.type = 'SegDataPreProcessor'

model = init_model(cfg, CHECKPOINT_FILE, device='cuda' if torch.cuda.is_available() else 'cpu')
model.dataset_meta = dict(classes=classes, palette=palette)

# === Inference
result = inference_model(model, IMAGE_PATH)

# === Visualization
visualizer = SegLocalVisualizer(
    vis_backends=[dict(type='LocalVisBackend')],
    save_dir='demo'
)
visualizer.dataset_meta = dict(classes=classes, palette=palette)
visualizer.add_datasample(
    name='result',
    image=cv2.imread(IMAGE_PATH),
    data_sample=result,
    draw_gt=False,
    show=False,
    wait_time=0,
    out_file=OUTPUT_PATH
)

print(f"Inference complete. Output saved to {OUTPUT_PATH}")

import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
RAW_IMAGE_DIR = 'mmsegmentation/dataset/images'       # Folder with RGB images (.jpg)
RAW_MASK_DIR = 'mmsegmentation/dataset/labels'        # Folder with segmentation masks (.png)
OUTPUT_ROOT = 'data/dataset'
IMAGE_SIZE = (512, 512)
TRAIN_SPLIT = 0.8

# === OUTPUT STRUCTURE ===
train_img_dir = os.path.join(OUTPUT_ROOT, 'img_dir/train')
val_img_dir = os.path.join(OUTPUT_ROOT, 'img_dir/val')
train_mask_dir = os.path.join(OUTPUT_ROOT, 'ann_dir/train')
val_mask_dir = os.path.join(OUTPUT_ROOT, 'ann_dir/val')

for path in [train_img_dir, val_img_dir, train_mask_dir, val_mask_dir]:
    os.makedirs(path, exist_ok=True)

# === LOAD AND SHUFFLE FILES ===
image_filenames = [f for f in os.listdir(RAW_IMAGE_DIR) if f.lower().endswith('.jpg')]
image_filenames.sort()
random.shuffle(image_filenames)

split_idx = int(len(image_filenames) * TRAIN_SPLIT)
train_files = image_filenames[:split_idx]
val_files = image_filenames[split_idx:]

# === PROCESS FUNCTION ===
def process_dataset(file_list, img_out_dir, mask_out_dir):
    for filename in tqdm(file_list, desc=f"Processing {img_out_dir}"):
        img_path = os.path.join(RAW_IMAGE_DIR, filename)
        mask_name = os.path.splitext(filename)[0] + '.png'
        mask_path = os.path.join(RAW_MASK_DIR, mask_name)

        if not os.path.exists(mask_path):
            print(f"⚠️ Mask not found for {filename}, skipping.")
            continue

        # Open and resize RGB image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize(IMAGE_SIZE, Image.BILINEAR)
        img_resized.save(os.path.join(img_out_dir, filename))

        # Open and resize mask using NEAREST (to preserve class values)
        mask = Image.open(mask_path)
        mask_resized = mask.resize(IMAGE_SIZE, Image.NEAREST)
        mask_resized.save(os.path.join(mask_out_dir, mask_name))

# === RUN ===
process_dataset(train_files, train_img_dir, train_mask_dir)
process_dataset(val_files, val_img_dir, val_mask_dir)

print("\n✅ Dataset successfully resized, split, and saved to:", OUTPUT_ROOT)

import os
from PIL import Image
import numpy as np

MASK_DIR = 'mmsegmentation/dataset/labels'  # path to your mask folder

unique_classes = set()

for filename in os.listdir(MASK_DIR):
    if filename.endswith('.png'):
        mask_path = os.path.join(MASK_DIR, filename)
        mask = Image.open(mask_path)
        mask_np = np.array(mask)
        unique_values = np.unique(mask_np)
        unique_classes.update(unique_values)

# Sort the class labels for readability
sorted_classes = sorted(list(unique_classes))

print("✅ Unique class values found in masks:", sorted_classes)
print("🧮 Total number of classes:", len(sorted_classes))

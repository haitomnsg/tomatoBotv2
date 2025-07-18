# /content/deep-hough-transform/data/split_and_format_filenames.py
import glob
import os
import random

# Define the directory where your original JPG images are
read_path = '/content/LineDetectionData' # Adjust as needed

# Define output paths
train_list_path = '/content/deep-hough-transform/data/training/agroNav_LineDetection_train.txt'
val_list_path = '/content/deep-hough-transform/data/training/agroNav_LineDetection_val.txt'

# Define the prefix for resized images
resized_prefix = '/content/deep-hough-transform/data/training/agroNav_LineDetection_resized_100_100/'

# Get all image filenames (without extension)
all_filenames = []
for file_path in glob.glob(os.path.join(read_path, '*.jpg')):
    filename = os.path.split(file_path)[1]
    filename_no_ext = os.path.splitext(filename)[0]
    all_filenames.append(filename_no_ext)

# Shuffle the filenames for random split
random.seed(42) # for reproducibility
random.shuffle(all_filenames)

# Define split ratio (e.g., 80% train, 20% val)
train_ratio = 0.8
train_size = int(len(all_filenames) * train_ratio)

train_filenames = all_filenames[:train_size]
val_filenames = all_filenames[train_size:]

print(f"Total images found: {len(all_filenames)}")
print(f"Training images: {len(train_filenames)}")
print(f"Validation images: {len(val_filenames)}")

# Write training list
with open(train_list_path, 'w') as f:
    for fname in train_filenames:
        f.write(f"{resized_prefix}{fname}\n")
        f.write(f"{resized_prefix}{fname}_flip\n") # Add flipped version for training

print(f"Training filename list created at: {train_list_path}")

# Write validation list
with open(val_list_path, 'w') as f:
    for fname in val_filenames:
        f.write(f"{resized_prefix}{fname}\n") # No flipped version for validation

print(f"Validation filename list created at: {val_list_path}")
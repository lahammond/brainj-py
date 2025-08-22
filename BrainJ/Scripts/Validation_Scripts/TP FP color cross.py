import os
import numpy as np
from tifffile import imread, imsave
from skimage.measure import regionprops
from skimage.draw import line

def draw_cross(image, center, size=2, color=(255, 0, 255)):
    y, x = map(int, center)
    h, w = image.shape[:2]
    for dy in range(-size, size + 1):
        yy = np.clip(y + dy, 0, h - 1)
        image[yy, x] = color
    for dx in range(-size, size + 1):
        xx = np.clip(x + dx, 0, w - 1)
        image[y, xx] = color

def generate_overlay(mask_dir, point_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(mask_dir) if f.endswith('.tif') or f.endswith('.tiff'))

    for fname in files:
        mask_path = os.path.join(mask_dir, fname)
        point_path = os.path.join(point_dir, fname)
        if not os.path.exists(point_path):
            continue

        labels = imread(mask_path)
        points = imread(point_path)
        points_coords = np.column_stack(np.nonzero(points))

        matched_labels = set()
        matched_points_idx = set()

        for idx, (y, x) in enumerate(points_coords):
            label_value = labels[y, x]
            if label_value != 0:
                matched_labels.add(label_value)
                matched_points_idx.add(idx)

        overlay = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

        for prop in regionprops(labels.astype(np.uint32)):
            label_id = prop.label
            centroid = prop.centroid

            if label_id in matched_labels:
                color = (0, 255, 0)  # Green = matched
            else:
                color = (255, 0, 255)  # Magenta = unmatched
            draw_cross(overlay, centroid, size=2, color=color)

        # False negatives (manual points not overlapping any label)
        for idx, (y, x) in enumerate(points_coords):
            if idx not in matched_points_idx:
                draw_cross(overlay, (y, x), size=2, color=(255, 255, 0))  # Yellow = missed manual

        imsave(os.path.join(output_dir, fname), overlay)


# Paths DAPI
mask_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\BrainJ Labels'
point_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\Point_Images'
output_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\TP FP Overlay'

#Paths cfos
mask_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\cfos\Labels_cfos'
point_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\cfos\Point_Images_C1_cfos - 620'
output_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\cfos\TP FP Overlay'

#Paths tdtomato
mask_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\tdtomato\Labels'
point_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\tdtomato\Point_Images_C2_tdtomato - 620'
output_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated\tdtomato\TP FP Overlay'

#make output
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


generate_overlay(mask_dir, point_dir, output_dir)

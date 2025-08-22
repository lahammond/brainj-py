import os
import numpy as np
import tifffile
from skimage.measure import regionprops
from collections import Counter
import pandas as pd

def evaluate_pair(mask_path, point_path):
    auto = tifffile.imread(mask_path)
    manual = tifffile.imread(point_path)

    manual_coords = np.column_stack(np.nonzero(manual))
    point_label_ids = auto[manual_coords[:, 0], manual_coords[:, 1]]

    label_props = regionprops(auto)
    detected_labels = set(p.label for p in label_props if p.label != 0)

    counts = Counter(point_label_ids)
    matched_labels = {lbl for lbl in counts if lbl != 0}
    unmatched_labels = detected_labels - matched_labels
    fn = np.sum(point_label_ids == 0)
    multi_point_labels = {lbl for lbl, count in counts.items() if lbl != 0 and count > 1}

    tp = len(matched_labels)
    fp = len(unmatched_labels)

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else np.nan

    return {
        'filename': os.path.basename(mask_path),
        'auto_total': len(detected_labels),
        'manual_total': len(manual_coords),
        'matched_detections': tp,
        'unmatched_detections': fp,
        'missed_manual_points': fn,
        'ambiguous_labels': len(multi_point_labels),
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def evaluate_all(mask_dir, point_dir, output_csv='results.csv'):
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.tif')]
    results = []

    for fname in mask_files:
        mask_path = os.path.join(mask_dir, fname)
        point_path = os.path.join(point_dir, fname)
        if not os.path.exists(point_path):
            continue
        result = evaluate_pair(mask_path, point_path)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


mask_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\BrainJ Labels'
point_dir = r'D:\Project_Data\BrainJ Datasets\BrainJ validation\Automated_DAPI\Point_Images'

df = evaluate_all(mask_dir, point_dir)
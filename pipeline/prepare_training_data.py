"""
Prepare Cellpose self-training data from v3 outputs.

Selects high-confidence (frame, mask) pairs as pseudo-ground-truth for
Cellpose fine-tuning. Filters applied:
  1. Exclude stubborn cells (ko1/4 95% empty, ko1/8 50% empty)
  2. Mask area >= 30% of per-cell median (excludes Cellpose fragments)
  3. IoU with previous frame >= 0.5 (excludes mask-jumping frames)

Output (Cellpose training-format directory):
  training_data/
    {folder}_{stem}_t{T:04d}_img.tif      # raw frame, uint16
    {folder}_{stem}_t{T:04d}_masks.tif    # binary mask, uint16
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

INPUT_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
TRAIN_DIR = CENTERED_DIR / 'training_data'

EXCLUDED_CELLS = {'ko1/4', 'ko1/8'}
MASK_AREA_RATIO_THRESHOLD = 0.30
IOU_NEIGHBOR_THRESHOLD = 0.5
FRAME_STRIDE = 10        # take every 10th frame to keep training set small (~500-600 frames)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    TRAIN_DIR.mkdir(exist_ok=True)
    print(f'[setup] Training output: {TRAIN_DIR}')
    print(f'[setup] Excluded cells: {EXCLUDED_CELLS}')

    counts = {'total_frames': 0, 'kept': 0,
              'excl_cell': 0, 'excl_empty': 0,
              'excl_area': 0, 'excl_iou_neighbor': 0}

    for folder in ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']:
        for raw_path in sorted((INPUT_DIR / folder).glob('*.tif')):
            stem = raw_path.stem
            cell_id = f'{folder}/{stem}'

            if cell_id in EXCLUDED_CELLS:
                counts['excl_cell'] += 1
                continue

            mask_path = CENTERED_DIR / folder / f'{stem}_mask.tif'
            if not mask_path.exists():
                continue

            raw = tifffile.imread(raw_path)
            mask = tifffile.imread(mask_path) > 0
            T = raw.shape[0]

            areas = mask.sum(axis=(1, 2))
            nonzero = areas[areas > 0]
            median_area = float(np.median(nonzero)) if len(nonzero) > 0 else 0.0
            area_threshold = median_area * MASK_AREA_RATIO_THRESHOLD

            for t in range(0, T, FRAME_STRIDE):
                counts['total_frames'] += 1

                if not mask[t].any():
                    counts['excl_empty'] += 1
                    continue

                if areas[t] < area_threshold:
                    counts['excl_area'] += 1
                    continue

                if t > 0 and mask[t - 1].any():
                    if iou(mask[t], mask[t - 1]) < IOU_NEIGHBOR_THRESHOLD:
                        counts['excl_iou_neighbor'] += 1
                        continue

                img_out = TRAIN_DIR / f'{folder}_{stem}_t{t:04d}_img.tif'
                mask_out = TRAIN_DIR / f'{folder}_{stem}_t{t:04d}_masks.tif'
                tifffile.imwrite(img_out, raw[t].astype(np.uint16))
                tifffile.imwrite(mask_out, mask[t].astype(np.uint16))
                counts['kept'] += 1

    print('\n[summary]')
    for k, v in counts.items():
        print(f'  {k:<25} {v}')
    print(f'\n[done] {counts["kept"]} (img, mask) pairs in {TRAIN_DIR}')


if __name__ == '__main__':
    main()

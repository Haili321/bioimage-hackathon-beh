"""
Prepare Cellpose training data using Quimp / biologist-validated reference masks.

Key alignment fix: most Quimp files end with `_centered_snakemask` -- the mask
was drawn on our centered.tif. To train a model that works on RAW frames
(matching the existing inference pipeline), we un-shift Quimp masks back to
raw frame using per-frame centroids from centering_manifest.json.

Output (Cellpose training-format directory, parallel to the existing one):
  Categorised_Data_centered/training_data_quimp/
    {folder}_{stem}_t{T:04d}_img.tif      raw frame, uint16
    {folder}_{stem}_t{T:04d}_masks.tif    binary mask, uint16, label = 1
"""
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import shift as ndshift

INPUT_DIR    = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
QUIMP_DIR    = Path('/dcs/pg25/u1898019/Desktop/Quimp_seg4CellPose_Retrain')
CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')   # has manifest with centers
TRAIN_DIR    = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered/training_data_quimp')

# 14 cells with Quimp segmentation
CELLS = [
    ('ki1', '12_centered_snakemask',  '12',   True),
    ('ki1', '23_centered_snakemask',  '23',   True),
    ('ki1', '25_centered_snakemask',  '25',   True),
    ('ki1', '3_centered_snakemask',   '3',    True),
    ('ki1', '7_centered_snakemask',   '7',    True),
    ('ko1', '22_centered_snakemask',  '22',   True),
    ('ko2', 'ko1_snakemask',          'ko1',  False),  # raw-frame
    ('ko2', 'ko11_centered_snakemask','ko11', True),
    ('ko2', 'ko30_snakemask',         'ko30', False),  # raw-frame
    ('wt1', '33_centered_snakemask',  '33',   True),
    ('wt1', '37_centered_snakemask',  '37',   True),
    ('wt1', '4_centered_snakemask',   '4',    True),
    ('wt2', 'wt14_centered_snakemask','wt14', True),
    ('wt2', 'wt4_centered_snakemask', 'wt4',  True),
]

FRAME_STRIDE = 5   # ~14 cells * 180 frames / 5 = ~500 pairs


def main():
    TRAIN_DIR.mkdir(exist_ok=True, parents=True)
    print(f'[setup] Training output: {TRAIN_DIR}')

    with open(CENTERED_DIR / 'centering_manifest.json') as f:
        manifest = json.load(f)

    counts = {'kept': 0, 'skipped_empty': 0, 'skipped_missing': 0}
    cell_stats = []

    for folder, qstem, ostem, is_centered in CELLS:
        qpath = QUIMP_DIR / folder / f'{qstem}.tif'
        raw_path = INPUT_DIR / folder / f'{ostem}.tif'
        manifest_key = f'{folder}/{ostem}.tif'

        if not qpath.exists() or not raw_path.exists():
            print(f'[miss] {folder}/{ostem}: quimp_exists={qpath.exists()}  raw_exists={raw_path.exists()}')
            counts['skipped_missing'] += 1
            continue

        raw = tifffile.imread(raw_path)
        quimp = tifffile.imread(qpath) > 0
        T, H, W = raw.shape
        assert quimp.shape == raw.shape, f'shape mismatch {quimp.shape} vs {raw.shape}'

        if is_centered and manifest_key in manifest:
            centers = np.array(manifest[manifest_key]['centers'])
        else:
            centers = None  # raw-frame Quimp, no shift needed

        per_cell_kept = 0
        for t in range(0, T, FRAME_STRIDE):
            qmask = quimp[t]
            if not qmask.any():
                counts['skipped_empty'] += 1
                continue

            # Un-shift Quimp mask back to raw frame
            if centers is not None:
                cy, cx = centers[t]
                # Forward (raw -> centered): shift = (H/2 - cy, W/2 - cx)
                # Inverse (centered -> raw): shift = (cy - H/2, cx - W/2)
                qmask_raw = ndshift(qmask.astype(np.uint8),
                                    shift=(cy - H / 2, cx - W / 2),
                                    order=0, mode='constant', cval=0).astype(bool)
            else:
                qmask_raw = qmask

            if not qmask_raw.any():
                counts['skipped_empty'] += 1
                continue

            img_out  = TRAIN_DIR / f'{folder}_{ostem}_t{t:04d}_img.tif'
            mask_out = TRAIN_DIR / f'{folder}_{ostem}_t{t:04d}_masks.tif'
            tifffile.imwrite(img_out,  raw[t].astype(np.uint16))
            tifffile.imwrite(mask_out, qmask_raw.astype(np.uint16))   # cellpose: label = 1
            counts['kept'] += 1
            per_cell_kept += 1

        cell_stats.append((f'{folder}/{ostem}', T, per_cell_kept))
        print(f'  {folder}/{ostem:<10} T={T:>3}  kept={per_cell_kept:>3}  centered={is_centered}')

    print(f'\n[summary]')
    for k, v in counts.items():
        print(f'  {k}: {v}')
    print(f'  cells included: {len(cell_stats)}')
    total_pairs = counts['kept']
    print(f'  total (image, mask) pairs: {total_pairs}')


if __name__ == '__main__':
    main()

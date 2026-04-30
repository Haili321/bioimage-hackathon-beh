"""
Extract per-frame body/lamellipodia metrics for every cell, using Edward's
percentile-based separator on top of the v3 Cellpose+centering output.

Default: percentile_body = 50 (Edward's original setting)
         brighter pixels labelled as "body", dimmer as "lamellipodia"

If the biology lead picks the opposite direction, just swap the column
labels in downstream analysis (no need to re-run).

Outputs (CSV):
    per_frame_metrics.csv:  one row per (cell, frame), all raw metrics
    per_cell_summary.csv:   one row per cell, time-aggregated
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

INPUT_DIR  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
OUT_DIR = CENTERED_DIR / 'trajectories'
OUT_DIR.mkdir(exist_ok=True)

PERCENTILE = 50   # Edward's default
ASSUMPTION = 'brighter_is_body'  # opposite would be 'brighter_is_lamellipodia'

print(f'[setup] percentile = {PERCENTILE}, assumption = {ASSUMPTION}')


def percentile_split(img: np.ndarray, mask: np.ndarray, p: float = PERCENTILE):
    """Returns (body_mask, lam_mask, threshold). Returns None if mask is empty."""
    pixels = img[mask]
    if len(pixels) == 0:
        return None
    thresh = float(np.percentile(pixels, p))
    body = (img >= thresh) & mask
    lam  = (img <  thresh) & mask
    return body, lam, thresh


def main():
    per_frame_records = []
    per_cell_summary  = []

    folders = ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']
    total_cells = 0
    total_frames = 0
    total_empty = 0

    for folder in folders:
        centered_files = sorted((CENTERED_DIR / folder).glob('*_centered.tif'))
        for centered_path in centered_files:
            stem = centered_path.stem.replace('_centered', '')
            mask_path = CENTERED_DIR / folder / f'{stem}_mask.tif'
            cell_id = f'{folder}/{stem}'

            centered = tifffile.imread(centered_path)
            mask = tifffile.imread(mask_path) > 0
            T = centered.shape[0]
            total_cells += 1

            cell_frame_records = []
            n_empty_in_cell = 0
            for t in range(T):
                img = centered[t].astype(np.float32)
                m = mask[t]
                total_frames += 1

                if m.sum() == 0:
                    n_empty_in_cell += 1
                    total_empty += 1
                    cell_frame_records.append({
                        'cell': cell_id,
                        'condition': folder[:2],
                        'category': folder[2:],
                        'batch': 'low' if folder.endswith('1') else 'high',
                        'frame': t,
                        'mask_area': 0,
                        'body_area': 0,
                        'lam_area': 0,
                        'body_mean_I': 0.0,
                        'lam_mean_I': 0.0,
                        'threshold': 0.0,
                        'empty': 1,
                    })
                    continue

                result = percentile_split(img, m)
                if result is None:
                    continue
                body, lam, thresh = result

                cell_frame_records.append({
                    'cell': cell_id,
                    'condition': folder[:2],
                    'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'frame': t,
                    'mask_area': int(m.sum()),
                    'body_area': int(body.sum()),
                    'lam_area': int(lam.sum()),
                    'body_mean_I': float(img[body].mean()) if body.any() else 0.0,
                    'lam_mean_I':  float(img[lam].mean())  if lam.any()  else 0.0,
                    'threshold': thresh,
                    'empty': 0,
                })

            per_frame_records.extend(cell_frame_records)

            # Per-cell summary (only over valid frames)
            valid = [r for r in cell_frame_records if not r['empty']]
            if valid:
                lam_areas = np.array([r['lam_area'] for r in valid])
                body_areas = np.array([r['body_area'] for r in valid])
                mask_areas = np.array([r['mask_area'] for r in valid])
                lam_Is = np.array([r['lam_mean_I'] for r in valid])
                body_Is = np.array([r['body_mean_I'] for r in valid])
                ratios = lam_areas / (body_areas + 1e-8)

                # Linear trend in lamellipodia area over time
                ts = np.array([r['frame'] for r in valid])
                if len(ts) >= 2:
                    slope_lam_area = float(np.polyfit(ts, lam_areas, 1)[0])
                    slope_body_area = float(np.polyfit(ts, body_areas, 1)[0])
                    slope_lam_I = float(np.polyfit(ts, lam_Is, 1)[0])
                else:
                    slope_lam_area = slope_body_area = slope_lam_I = 0.0

                per_cell_summary.append({
                    'cell': cell_id,
                    'condition': folder[:2],
                    'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'T': T,
                    'n_valid_frames': len(valid),
                    'n_empty_frames': n_empty_in_cell,
                    'mean_mask_area': float(mask_areas.mean()),
                    'mean_body_area': float(body_areas.mean()),
                    'mean_lam_area':  float(lam_areas.mean()),
                    'mean_body_I':    float(body_Is.mean()),
                    'mean_lam_I':     float(lam_Is.mean()),
                    'mean_lam_to_body_ratio': float(ratios.mean()),
                    'slope_lam_area_per_frame':  slope_lam_area,
                    'slope_body_area_per_frame': slope_body_area,
                    'slope_lam_I_per_frame':     slope_lam_I,
                })
                print(f'  {cell_id:<24} T={T:>3} valid={len(valid):>3} '
                      f'mean_lam_area={lam_areas.mean():>6.0f} mean_body_area={body_areas.mean():>6.0f} '
                      f'lam/body={ratios.mean():.3f}')

    # Write CSVs
    per_frame_path = OUT_DIR / 'per_frame_metrics.csv'
    per_cell_path  = OUT_DIR / 'per_cell_summary.csv'

    if per_frame_records:
        with open(per_frame_path, 'w', newline='') as f:
            fieldnames = list(per_frame_records[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_frame_records)
        print(f'\n[done] {per_frame_path}: {len(per_frame_records)} rows')

    if per_cell_summary:
        with open(per_cell_path, 'w', newline='') as f:
            fieldnames = list(per_cell_summary[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_cell_summary)
        print(f'[done] {per_cell_path}: {len(per_cell_summary)} rows')

    print(f'\n[summary] cells={total_cells}, frames={total_frames}, '
          f'empty={total_empty} ({total_empty/total_frames*100:.2f}%)')


if __name__ == '__main__':
    main()

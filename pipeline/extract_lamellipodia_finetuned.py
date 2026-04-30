"""
Per-cell lamellipodia / cytoplasm metrics using a percentile split inside
the fine-tuned Cellpose masks, applied across all 30 cells.

Convention (confirmed by biology lead):
  brighter pixels = lamellipodia (Arp2/3 concentrated at the leading edge)
  dimmer pixels   = cytoplasmic Arp2/3 pool

Default: percentile = 50  (top 50% = lamellipodia, bottom 50% = cytoplasm)

Outputs:
  per_frame_metrics.csv  one row per (cell, frame)
  per_cell_summary.csv   one row per cell, time-aggregated
"""
import csv
from pathlib import Path

import numpy as np
import tifffile

CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
OUT_DIR = CENTERED_DIR / 'lamellipodia'
OUT_DIR.mkdir(exist_ok=True)

PERCENTILE = 50


def main():
    folders = ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']
    per_frame = []
    per_cell = []

    for folder in folders:
        for centered_path in sorted((CENTERED_DIR / folder).glob('*_centered.tif')):
            stem = centered_path.stem.replace('_centered', '')
            mask_path = CENTERED_DIR / folder / f'{stem}_mask.tif'
            cell_id = f'{folder}/{stem}'

            centered = tifffile.imread(centered_path)
            mask = tifffile.imread(mask_path) > 0
            T = centered.shape[0]

            cell_frame_records = []
            for t in range(T):
                img = centered[t].astype(np.float32)
                m = mask[t]
                if m.sum() == 0:
                    continue
                pixels = img[m]
                if len(pixels) == 0:
                    continue
                thresh = float(np.percentile(pixels, PERCENTILE))

                # Brighter half = lamellipodia, dimmer half = cytoplasm
                lam = (img >= thresh) & m
                cyto = (img < thresh) & m

                lam_pixels = img[lam] if lam.any() else np.array([])
                cyto_pixels = img[cyto] if cyto.any() else np.array([])

                lam_mean = float(lam_pixels.mean()) if len(lam_pixels) > 0 else 0.0
                cyto_mean = float(cyto_pixels.mean()) if len(cyto_pixels) > 0 else 0.0

                cell_frame_records.append({
                    'cell': cell_id,
                    'condition': folder[:2],
                    'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'frame': t,
                    'mask_area': int(m.sum()),
                    'lam_area': int(lam.sum()),
                    'cyto_area': int(cyto.sum()),
                    'lam_mean_I': lam_mean,
                    'cyto_mean_I': cyto_mean,
                    'lam_to_cyto_I_ratio': lam_mean / cyto_mean if cyto_mean > 0 else 0.0,
                    'threshold': thresh,
                })

            per_frame.extend(cell_frame_records)

            if cell_frame_records:
                lam_areas = np.array([r['lam_area'] for r in cell_frame_records])
                cyto_areas = np.array([r['cyto_area'] for r in cell_frame_records])
                lam_Is = np.array([r['lam_mean_I'] for r in cell_frame_records])
                cyto_Is = np.array([r['cyto_mean_I'] for r in cell_frame_records])
                ratios = np.array([r['lam_to_cyto_I_ratio'] for r in cell_frame_records])
                thrs = np.array([r['threshold'] for r in cell_frame_records])

                per_cell.append({
                    'cell': cell_id,
                    'condition': folder[:2],
                    'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'T': T,
                    'n_valid_frames': len(cell_frame_records),
                    'mean_mask_area': float(np.mean([r['mask_area'] for r in cell_frame_records])),
                    'mean_lam_area': float(lam_areas.mean()),
                    'mean_cyto_area': float(cyto_areas.mean()),
                    'mean_lam_I': float(lam_Is.mean()),
                    'mean_cyto_I': float(cyto_Is.mean()),
                    'mean_lam_to_cyto_I_ratio': float(ratios.mean()),
                    'std_lam_to_cyto_I_ratio': float(ratios.std()),
                    'mean_threshold': float(thrs.mean()),
                })
                s = per_cell[-1]
                print(f'  {cell_id:<25} N={len(cell_frame_records):>3}  '
                      f'lam/cyto_I = {s["mean_lam_to_cyto_I_ratio"]:.3f} ± {s["std_lam_to_cyto_I_ratio"]:.3f}  '
                      f'lam_I = {s["mean_lam_I"]:>5.0f}  cyto_I = {s["mean_cyto_I"]:>5.0f}')

    # Save CSVs
    with open(OUT_DIR / 'per_frame_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)
    with open(OUT_DIR / 'per_cell_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_cell[0].keys()))
        writer.writeheader()
        writer.writerows(per_cell)

    print(f'\n[done] per_frame: {len(per_frame)}, per_cell: {len(per_cell)}')

    # Per-condition summary
    print('\n=== Per-condition summary (fine-tuned masks, brighter = lamellipodia) ===')
    for cond in ['wt', 'ko', 'ki']:
        rows = [r for r in per_cell if r['condition'] == cond]
        if not rows: continue
        ratios = np.array([r['mean_lam_to_cyto_I_ratio'] for r in rows])
        lam_Is = np.array([r['mean_lam_I'] for r in rows])
        cyto_Is = np.array([r['mean_cyto_I'] for r in rows])
        print(f'  {cond.upper()} (N={len(rows)}): '
              f'lam/cyto_I = {ratios.mean():.3f} ± {ratios.std():.3f}  '
              f'lam_I = {lam_Is.mean():>5.0f} ± {lam_Is.std():>4.0f}  '
              f'cyto_I = {cyto_Is.mean():>5.0f} ± {cyto_Is.std():>4.0f}')


if __name__ == '__main__':
    main()

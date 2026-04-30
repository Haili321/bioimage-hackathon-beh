"""
Lamellipodia / cytoplasm metrics under the Quimp-trained model.

Runs on BOTH batches. Same percentile=50 split logic.
"""
import csv
from pathlib import Path

import numpy as np
import tifffile

JOBS = [
    Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned'),
    Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned'),
]
PERCENTILE = 50


for centered_dir in JOBS:
    if not (centered_dir / 'centering_manifest.json').exists():
        print(f'[skip] {centered_dir.name}: manifest missing')
        continue
    out_dir = centered_dir / 'lamellipodia'
    out_dir.mkdir(exist_ok=True)
    print(f'\n=== {centered_dir.name} ===')

    folders = ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']
    per_frame, per_cell = [], []

    for folder in folders:
        for centered_path in sorted((centered_dir / folder).glob('*_centered.tif')):
            stem = centered_path.stem.replace('_centered', '')
            mask_path = centered_dir / folder / f'{stem}_mask.tif'
            cell_id = f'{folder}/{stem}'
            centered = tifffile.imread(centered_path)
            mask = tifffile.imread(mask_path) > 0
            T = centered.shape[0]

            cell_records = []
            for t in range(T):
                img = centered[t].astype(np.float32)
                m = mask[t]
                if m.sum() == 0:
                    continue
                pixels = img[m]
                if len(pixels) == 0:
                    continue
                thresh = float(np.percentile(pixels, PERCENTILE))
                lam = (img >= thresh) & m
                cyto = (img < thresh) & m
                lam_pixels = img[lam] if lam.any() else np.array([])
                cyto_pixels = img[cyto] if cyto.any() else np.array([])
                lam_mean = float(lam_pixels.mean()) if len(lam_pixels) > 0 else 0.0
                cyto_mean = float(cyto_pixels.mean()) if len(cyto_pixels) > 0 else 0.0
                cell_records.append({
                    'cell': cell_id, 'condition': folder[:2], 'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'frame': t, 'mask_area': int(m.sum()),
                    'lam_mean_I': lam_mean, 'cyto_mean_I': cyto_mean,
                    'lam_to_cyto_I_ratio': lam_mean / cyto_mean if cyto_mean > 0 else 0.0,
                    'threshold': thresh,
                })
            per_frame.extend(cell_records)
            if cell_records:
                ratios = np.array([r['lam_to_cyto_I_ratio'] for r in cell_records])
                lam_Is = np.array([r['lam_mean_I']     for r in cell_records])
                cyto_Is = np.array([r['cyto_mean_I']    for r in cell_records])
                per_cell.append({
                    'cell': cell_id, 'condition': folder[:2], 'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'T': T, 'n_valid_frames': len(cell_records),
                    'mean_lam_I': float(lam_Is.mean()),
                    'mean_cyto_I': float(cyto_Is.mean()),
                    'mean_lam_to_cyto_I_ratio': float(ratios.mean()),
                    'std_lam_to_cyto_I_ratio': float(ratios.std()),
                })

    with open(out_dir / 'per_cell_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_cell[0].keys())); w.writeheader(); w.writerows(per_cell)
    print(f'[done] {len(per_cell)} cells -> {out_dir / "per_cell_summary.csv"}')

    print('Per-condition lam/cyto ratio:')
    for cond in ['wt', 'ko', 'ki']:
        rows = [r for r in per_cell if r['condition'] == cond]
        ratios = np.array([r['mean_lam_to_cyto_I_ratio'] for r in rows])
        print(f'  {cond.upper()} (N={len(rows)}): {ratios.mean():.3f} ± {ratios.std():.3f}')

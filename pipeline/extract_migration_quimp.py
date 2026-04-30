"""
Migration metrics with the Quimp-trained model. Same logic as
extract_migration_finetuned.py / _2nd.py, just different paths.

Runs on BOTH batches.
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

JOBS = [
    Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned'),
    Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned'),
]

PIXEL_SIZE_UM = 0.318
FRAME_INTERVAL_MIN = 1.0
AREA_RATIO_THRESHOLD = 0.30
IOU_THRESHOLD = 0.30


def iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


for centered_dir in JOBS:
    if not (centered_dir / 'centering_manifest.json').exists():
        print(f'[skip] {centered_dir.name}: manifest missing')
        continue
    out_dir = centered_dir / 'trajectories'
    out_dir.mkdir(exist_ok=True)
    print(f'\n=== {centered_dir.name} ===')

    with open(centered_dir / 'centering_manifest.json') as f:
        manifest = json.load(f)

    records = []
    trajectories = {}
    for rel_path, info in manifest.items():
        centers = np.array(info['centers'], dtype=np.float64)
        T = info['shape'][0]
        folder = rel_path.split('/')[0]
        cond = folder[:2]
        cat = folder[2:]
        batch = 'low' if cat == '1' else 'high'
        cell_id = rel_path.replace('.tif', '')
        stem = Path(rel_path).stem
        mask_path = centered_dir / folder / f'{stem}_mask.tif'
        mask_stack = tifffile.imread(mask_path) > 0
        areas = mask_stack.sum(axis=(1, 2))
        nonzero_areas = areas[areas > 0]
        median_area = float(np.median(nonzero_areas)) if len(nonzero_areas) > 0 else 0.0
        area_threshold = median_area * AREA_RATIO_THRESHOLD
        valid_frames = (areas > 0) & (areas >= area_threshold)
        pair_valid = np.zeros(T - 1, dtype=bool)
        ious = np.zeros(T - 1)
        for t in range(T - 1):
            if valid_frames[t] and valid_frames[t + 1]:
                ious[t] = iou(mask_stack[t], mask_stack[t + 1])
                pair_valid[t] = ious[t] >= IOU_THRESHOLD
        deltas = np.diff(centers, axis=0)
        speeds_px = np.linalg.norm(deltas, axis=1)
        speeds_um = speeds_px * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN
        n_pairs = int(pair_valid.sum())
        if n_pairs > 0:
            sp = speeds_um[pair_valid]
            path_um = float(speeds_px[pair_valid].sum() * PIXEL_SIZE_UM)
            valid_idx = np.where(valid_frames)[0]
            first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
            net_um = float(np.linalg.norm(centers[last_v] - centers[first_v]) * PIXEL_SIZE_UM)
            persistence = min(net_um / max(path_um, 1e-8), 1.0)
            mean_speed = float(sp.mean()); median_speed = float(np.median(sp)); max_speed = float(sp.max()); std_speed = float(sp.std())
        else:
            path_um = net_um = persistence = mean_speed = median_speed = max_speed = std_speed = 0.0

        records.append({
            'cell': cell_id, 'condition': cond, 'category': cat, 'batch': batch,
            'T': T, 'n_valid_frames': int(valid_frames.sum()), 'n_valid_pairs': n_pairs,
            'mean_speed_um_per_min': mean_speed, 'median_speed_um_per_min': median_speed,
            'max_speed_um_per_min': max_speed, 'std_speed': std_speed,
            'total_path_um': path_um, 'net_disp_um': net_um, 'persistence': persistence,
        })

    with open(out_dir / 'migration_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys())); w.writeheader(); w.writerows(records)
    print(f'[done] {len(records)} cells -> {out_dir / "migration_metrics.csv"}')

    print('Per-condition:')
    for cond in ['wt', 'ko', 'ki']:
        cr = [r for r in records if r['condition'] == cond]
        sp = np.array([r['mean_speed_um_per_min'] for r in cr])
        print(f'  {cond.upper()} (N={len(cr)}): mean_speed = {sp.mean():.2f} ± {sp.std():.2f}')

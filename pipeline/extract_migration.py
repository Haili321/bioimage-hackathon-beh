"""
Compute cell-migration metrics from per-frame centroids saved during the
v3 pipeline (centering_manifest.json).

Per-cell metrics:
    mean_speed_um_per_min, max_speed_um_per_min, std_speed
    total_path_um            (cumulative travel distance)
    net_displacement_um      (straight line, start to end)
    persistence              (net / total, 1=straight, 0=random)
    duration_min

Two versions are reported:
    *_all      include all frames (empty-mask frames use the last known
               centroid, which deflates speed for cells with many empty masks)
    *_valid    only frames with non-empty Cellpose masks contribute
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
OUT_DIR = CENTERED_DIR / 'trajectories'
OUT_DIR.mkdir(exist_ok=True)

PIXEL_SIZE_UM = 0.318
FRAME_INTERVAL_MIN = 1.0  # 60 seconds

with open(CENTERED_DIR / 'centering_manifest.json') as f:
    manifest = json.load(f)

print(f'[setup] {len(manifest)} cells in manifest')
print(f'[setup] pixel size {PIXEL_SIZE_UM} um/px, frame interval {FRAME_INTERVAL_MIN} min')

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

    mask_path = CENTERED_DIR / folder / f'{stem}_mask.tif'
    mask_stack = tifffile.imread(mask_path)
    valid_frames = np.array([mask_stack[t].sum() > 0 for t in range(T)])

    deltas = np.diff(centers, axis=0)
    speeds_px_per_frame = np.linalg.norm(deltas, axis=1)
    speeds_um_per_min = speeds_px_per_frame * PIXEL_SIZE_UM / FRAME_INTERVAL_MIN

    valid_pairs = valid_frames[:-1] & valid_frames[1:]
    n_valid_pairs = int(valid_pairs.sum())

    total_path_all = float(speeds_px_per_frame.sum() * PIXEL_SIZE_UM)
    net_disp_all = float(np.linalg.norm(centers[-1] - centers[0]) * PIXEL_SIZE_UM)
    persistence_all = net_disp_all / max(total_path_all, 1e-8)

    if n_valid_pairs > 0:
        s_valid = speeds_um_per_min[valid_pairs]
        total_path_valid = float(speeds_px_per_frame[valid_pairs].sum() * PIXEL_SIZE_UM)
        valid_idx = np.where(valid_frames)[0]
        first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
        net_disp_valid = float(np.linalg.norm(centers[last_v] - centers[first_v]) * PIXEL_SIZE_UM)
        persistence_valid = net_disp_valid / max(total_path_valid, 1e-8)
        mean_speed_v = float(s_valid.mean())
        max_speed_v = float(s_valid.max())
        std_speed_v = float(s_valid.std())
    else:
        total_path_valid = net_disp_valid = persistence_valid = 0.0
        mean_speed_v = max_speed_v = std_speed_v = 0.0

    rec = {
        'cell': cell_id,
        'condition': cond,
        'category': cat,
        'batch': batch,
        'T': T,
        'duration_min': (T - 1) * FRAME_INTERVAL_MIN,
        'n_valid_frames': int(valid_frames.sum()),
        'n_valid_pairs': n_valid_pairs,
        # All-frame versions
        'mean_speed_um_per_min_all': float(speeds_um_per_min.mean()),
        'max_speed_um_per_min_all': float(speeds_um_per_min.max()),
        'std_speed_all': float(speeds_um_per_min.std()),
        'total_path_um_all': total_path_all,
        'net_disp_um_all': net_disp_all,
        'persistence_all': persistence_all,
        # Valid-only versions
        'mean_speed_um_per_min_valid': mean_speed_v,
        'max_speed_um_per_min_valid': max_speed_v,
        'std_speed_valid': std_speed_v,
        'total_path_um_valid': total_path_valid,
        'net_disp_um_valid': net_disp_valid,
        'persistence_valid': persistence_valid,
    }
    records.append(rec)

    traj_um = (centers - centers[0]) * PIXEL_SIZE_UM
    trajectories[cell_id] = {
        'traj_um': traj_um.tolist(),
        'valid': valid_frames.tolist(),
        'condition': cond,
        'category': cat,
        'batch': batch,
        'folder': folder,
    }

    print(f'  {cell_id:<24} T={T:>3} valid={int(valid_frames.sum()):>3} '
          f'speed={mean_speed_v:>5.2f}μm/min path={total_path_valid:>6.1f}μm '
          f'net={net_disp_valid:>5.1f}μm pers={persistence_valid:.3f}')

# CSV
csv_path = OUT_DIR / 'migration_metrics.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
print(f'\n[done] {csv_path}: {len(records)} rows')

# Save trajectories JSON for visualization
with open(OUT_DIR / 'trajectories.json', 'w') as f:
    json.dump(trajectories, f)
print(f'[done] {OUT_DIR / "trajectories.json"}')

# Per-condition summary
print('\n=== Per-condition summary (mean ± std, valid-only) ===')
for cond in ['wt', 'ko', 'ki']:
    cond_records = [r for r in records if r['condition'] == cond]
    speeds = np.array([r['mean_speed_um_per_min_valid'] for r in cond_records])
    paths = np.array([r['total_path_um_valid'] for r in cond_records])
    nets = np.array([r['net_disp_um_valid'] for r in cond_records])
    pers = np.array([r['persistence_valid'] for r in cond_records])
    print(f'  {cond.upper()} (N={len(cond_records)}): '
          f'speed={speeds.mean():>4.2f}±{speeds.std():>4.2f}μm/min  '
          f'path={paths.mean():>5.0f}±{paths.std():>5.0f}μm  '
          f'net={nets.mean():>4.0f}±{nets.std():>4.0f}μm  '
          f'pers={pers.mean():.2f}±{pers.std():.2f}')

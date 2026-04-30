"""
Compute cell-migration metrics from per-frame centroids saved during the
v3 pipeline (centering_manifest.json), with mask-stability filtering.

A frame pair is included in migration computations only if:
  1. Both frames have non-trivial mask area (>= 30% of the per-cell median).
  2. Consecutive masks overlap by at least IoU >= 0.3.

This filter rejects Cellpose fragmentation artefacts (cell occasionally
detected as a small fragment for one frame, causing centroid jumps).
Discovered while analysing wt1/33 — see demos/wt1_33_mask_diagnostic.png.

Per-cell metrics:
    mean / median / max speed (μm/min)
    total path length        (cumulative travel)
    net displacement         (straight line, start to end)
    persistence              (= net / total, 1=straight, 0=random)
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile

CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
OUT_DIR = CENTERED_DIR / 'trajectories'
OUT_DIR.mkdir(exist_ok=True)

PIXEL_SIZE_UM = 0.318
FRAME_INTERVAL_MIN = 1.0

AREA_RATIO_THRESHOLD = 0.30  # frame's mask area must be >= 30% of per-cell median
IOU_THRESHOLD = 0.30          # consecutive masks must overlap by >= 30%


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


with open(CENTERED_DIR / 'centering_manifest.json') as f:
    manifest = json.load(f)

print(f'[setup] {len(manifest)} cells')
print(f'[setup] filter: mask area >= 30% per-cell median  AND  IoU >= 0.3')

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
    mask_stack = tifffile.imread(mask_path) > 0
    areas = mask_stack.sum(axis=(1, 2))

    # Per-cell adaptive area threshold
    nonzero_areas = areas[areas > 0]
    median_area = float(np.median(nonzero_areas)) if len(nonzero_areas) > 0 else 0.0
    area_threshold = median_area * AREA_RATIO_THRESHOLD

    # Frame validity: non-empty AND area >= threshold
    valid_frames = (areas > 0) & (areas >= area_threshold)

    # Pair validity: both frames valid AND IoU >= threshold
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
    n_rejected_area = int(~valid_frames.sum())
    n_rejected_iou = int((valid_frames[:-1] & valid_frames[1:] & (ious < IOU_THRESHOLD)).sum())

    if n_pairs > 0:
        sp = speeds_um[pair_valid]
        path_um = float(speeds_px[pair_valid].sum() * PIXEL_SIZE_UM)
        # Net displacement: between first and last valid frame
        valid_idx = np.where(valid_frames)[0]
        first_v, last_v = int(valid_idx[0]), int(valid_idx[-1])
        net_um = float(np.linalg.norm(centers[last_v] - centers[first_v]) * PIXEL_SIZE_UM)
        persistence = min(net_um / max(path_um, 1e-8), 1.0)  # cap at 1 (geometric upper bound)
        mean_speed = float(sp.mean())
        median_speed = float(np.median(sp))
        max_speed = float(sp.max())
        std_speed = float(sp.std())
    else:
        sp = np.array([])
        path_um = net_um = persistence = 0.0
        mean_speed = median_speed = max_speed = std_speed = 0.0

    rec = {
        'cell': cell_id,
        'condition': cond,
        'category': cat,
        'batch': batch,
        'T': T,
        'duration_min': (T - 1) * FRAME_INTERVAL_MIN,
        'median_mask_area': int(median_area),
        'area_threshold': int(area_threshold),
        'n_valid_frames': int(valid_frames.sum()),
        'n_valid_pairs': n_pairs,
        'n_rejected_area_frames': int((~valid_frames).sum()),
        'n_rejected_iou_pairs': int(n_rejected_iou),
        'mean_speed_um_per_min': mean_speed,
        'median_speed_um_per_min': median_speed,
        'max_speed_um_per_min': max_speed,
        'std_speed': std_speed,
        'total_path_um': path_um,
        'net_disp_um': net_um,
        'persistence': persistence,
    }
    records.append(rec)

    # Save trajectory (only valid frames, for plotting)
    traj_um = (centers - centers[valid_idx[0]] if n_pairs > 0 else centers - centers[0]) * PIXEL_SIZE_UM
    trajectories[cell_id] = {
        'traj_um': traj_um.tolist(),
        'valid': valid_frames.tolist(),
        'pair_valid': pair_valid.tolist(),
        'condition': cond,
        'category': cat,
        'batch': batch,
        'folder': folder,
    }

    print(f'  {cell_id:<24} T={T:>3} valid={int(valid_frames.sum()):>3} pairs={n_pairs:>3} '
          f'(rej {n_rejected_iou} for low IoU) '
          f'speed={mean_speed:>5.2f}μm/min med={median_speed:>4.2f} '
          f'path={path_um:>5.0f}μm net={net_um:>4.0f}μm pers={persistence:.3f}')

# CSV
with open(OUT_DIR / 'migration_metrics.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
print(f'\n[done] migration_metrics.csv ({len(records)} rows)')

with open(OUT_DIR / 'trajectories.json', 'w') as f:
    json.dump(trajectories, f)

print('\n=== Per-condition summary (filter applied: area>=30% median, IoU>=0.3) ===')
for cond in ['wt', 'ko', 'ki']:
    cr = [r for r in records if r['condition'] == cond]
    sp_mean = np.array([r['mean_speed_um_per_min'] for r in cr])
    sp_med = np.array([r['median_speed_um_per_min'] for r in cr])
    paths = np.array([r['total_path_um'] for r in cr])
    nets = np.array([r['net_disp_um'] for r in cr])
    pers = np.array([r['persistence'] for r in cr])
    print(f'  {cond.upper()} (N={len(cr)}): '
          f'mean_speed={sp_mean.mean():.2f}±{sp_mean.std():.2f}  '
          f'median_speed={sp_med.mean():.2f}±{sp_med.std():.2f}  '
          f'path={paths.mean():.0f}±{paths.std():.0f}  '
          f'net={nets.mean():.0f}±{nets.std():.0f}  '
          f'pers={pers.mean():.2f}±{pers.std():.2f}')

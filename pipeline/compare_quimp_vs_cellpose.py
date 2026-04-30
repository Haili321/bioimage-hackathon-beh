"""
Compare Quimp active-contour reference masks (Fiji + biologist-supervised)
against Cellpose masks for the 14 cells where a Quimp segmentation exists.

This is read-only on existing files. Outputs go to a NEW directory only.

Usage:
    python compare_quimp_vs_cellpose.py finetuned    # vs fine-tuned cyto3
    python compare_quimp_vs_cellpose.py v3           # vs zero-shot cyto3 (v3 baseline)

Output:
    <Categorised_Data>/quimp_vs_<target>/
        comparison_per_frame.csv    # row per (cell, frame)
        comparison_summary.csv      # row per cell
        per_condition_summary.csv   # row per condition (wt/ko/ki)
"""
import csv
import sys
from pathlib import Path

import numpy as np

# Import the helpers from compare_masks.py without modifying it
sys.path.insert(0, str(Path(__file__).parent))
from compare_masks import compare_stacks, _load_mask_stack

QUIMP_DIR = Path('/dcs/pg25/u1898019/Desktop/Quimp_seg4CellPose_Retrain')

TARGETS = {
    'finetuned': Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned'),
    'v3':        Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered'),
}

# 14 hand-seg cells.  Tuple is (folder, quimp_stem, our_stem).
# Some hand-seg files use *_centered_snakemask.tif, some *_snakemask.tif.
HAND_SEG_CELLS = [
    ('ki1', '12_centered_snakemask',  '12'),
    ('ki1', '23_centered_snakemask',  '23'),
    ('ki1', '25_centered_snakemask',  '25'),
    ('ki1', '3_centered_snakemask',   '3'),
    ('ki1', '7_centered_snakemask',   '7'),
    ('ko1', '22_centered_snakemask',  '22'),
    ('ko2', 'ko1_snakemask',          'ko1'),
    ('ko2', 'ko11_centered_snakemask','ko11'),
    ('ko2', 'ko30_snakemask',         'ko30'),
    ('wt1', '33_centered_snakemask',  '33'),
    ('wt1', '37_centered_snakemask',  '37'),
    ('wt1', '4_centered_snakemask',   '4'),
    ('wt2', 'wt14_centered_snakemask','wt14'),
    ('wt2', 'wt4_centered_snakemask', 'wt4'),
]


def run(target: str):
    if target not in TARGETS:
        raise SystemExit(f'target must be one of {list(TARGETS)}')
    our_dir = TARGETS[target]
    out_dir = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data') / f'quimp_vs_{target}'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[setup] target = {target}')
    print(f'[setup] our masks dir = {our_dir}')
    print(f'[setup] output dir    = {out_dir}\n')

    all_records = []
    summaries = []

    for folder, quimp_stem, our_stem in HAND_SEG_CELLS:
        quimp_path = QUIMP_DIR / folder / f'{quimp_stem}.tif'
        our_path = our_dir / folder / f'{our_stem}_mask.tif'

        if not quimp_path.exists():
            print(f'[skip] {folder}/{our_stem}: quimp file not found at {quimp_path}')
            continue
        if not our_path.exists():
            print(f'[skip] {folder}/{our_stem}: our mask not found at {our_path}')
            continue

        a = _load_mask_stack(quimp_path)
        b = _load_mask_stack(our_path)

        if a.shape != b.shape:
            print(f'[skip] {folder}/{our_stem}: shape mismatch  quimp={a.shape}  ours={b.shape}')
            continue

        cell_id = f'{folder}/{our_stem}'
        records = compare_stacks(a, b, cell_id=cell_id)
        all_records.extend(records)

        # Per-cell summary, only frames where both masks are non-empty
        valid = [r for r in records if not np.isnan(r['centroid_dist_px'])]
        if not valid:
            print(f'[warn] {cell_id}: no frames with both masks non-empty')
            continue

        cond = folder[:2]
        batch = 'low' if folder[2:] == '1' else 'high'
        s = {
            'cell': cell_id,
            'condition': cond,
            'batch': batch,
            'n_frames': len(records),
            'n_valid': len(valid),
            'mean_iou':           float(np.mean([r['iou']           for r in valid])),
            'median_iou':         float(np.median([r['iou']         for r in valid])),
            'mean_dice':          float(np.mean([r['dice']          for r in valid])),
            'mean_centroid_dist': float(np.mean([r['centroid_dist_px'] for r in valid])),
            'mean_boundary_iou':  float(np.mean([r['boundary_iou']  for r in valid])),
            'mean_hd95':          float(np.nanmean([r['hausdorff_95_px'] for r in valid])),
            'mean_area_ratio':    float(np.mean([r['area_ratio']    for r in valid])),
        }
        summaries.append(s)
        print(f'  {cell_id:<14} N={s["n_valid"]:>3}/{s["n_frames"]:<3} '
              f'IoU={s["mean_iou"]:.3f}  Dice={s["mean_dice"]:.3f}  '
              f'centroid={s["mean_centroid_dist"]:>4.1f}px  HD95={s["mean_hd95"]:>5.1f}px')

    # Write per-frame CSV
    if all_records:
        keys = list(all_records[0].keys())
        with open(out_dir / 'comparison_per_frame.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_records)
        print(f'\n[saved] {out_dir / "comparison_per_frame.csv"}  ({len(all_records)} rows)')

    # Write per-cell summary
    if summaries:
        keys = list(summaries[0].keys())
        with open(out_dir / 'comparison_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summaries)
        print(f'[saved] {out_dir / "comparison_summary.csv"}  ({len(summaries)} rows)')

    # Per-condition aggregation
    print(f'\n=== Per-condition summary ({target} vs Quimp) ===')
    cond_rows = []
    for cond in ['wt', 'ko', 'ki']:
        rows = [r for r in summaries if r['condition'] == cond]
        if not rows:
            continue
        iou  = np.array([r['mean_iou']           for r in rows])
        dice = np.array([r['mean_dice']          for r in rows])
        cd   = np.array([r['mean_centroid_dist'] for r in rows])
        hd95 = np.array([r['mean_hd95']          for r in rows])
        biou = np.array([r['mean_boundary_iou']  for r in rows])
        ar   = np.array([r['mean_area_ratio']    for r in rows])
        cond_rows.append({
            'condition': cond, 'n_cells': len(rows),
            'mean_iou':           float(iou.mean()),
            'std_iou':            float(iou.std()),
            'mean_dice':          float(dice.mean()),
            'mean_boundary_iou':  float(biou.mean()),
            'mean_centroid_dist': float(cd.mean()),
            'mean_hd95':          float(hd95.mean()),
            'mean_area_ratio':    float(ar.mean()),
        })
        print(f'  {cond.upper()} (N={len(rows)}): '
              f'IoU={iou.mean():.3f}±{iou.std():.3f}  '
              f'Dice={dice.mean():.3f}  '
              f'b-IoU={biou.mean():.3f}  '
              f'centroid={cd.mean():.1f}px  '
              f'HD95={hd95.mean():.1f}px  '
              f'area_ratio={ar.mean():.3f}')

    # All-cell overall
    if summaries:
        all_iou = np.array([r['mean_iou']  for r in summaries])
        all_dice = np.array([r['mean_dice'] for r in summaries])
        all_hd95 = np.array([r['mean_hd95'] for r in summaries])
        print(f'\n  ALL  (N={len(summaries)}): '
              f'IoU={all_iou.mean():.3f}±{all_iou.std():.3f}  '
              f'Dice={all_dice.mean():.3f}  '
              f'HD95={all_hd95.mean():.1f}px')

    if cond_rows:
        keys = list(cond_rows[0].keys())
        with open(out_dir / 'per_condition_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(cond_rows)
        print(f'\n[saved] {out_dir / "per_condition_summary.csv"}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1])

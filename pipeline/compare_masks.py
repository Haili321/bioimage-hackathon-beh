"""
Generic mask-comparison utility.

Takes two parallel directory trees of paired *_mask.tif files (or raw .tif
that the helpers convert to masks), and computes per-frame comparison
metrics between every paired stack.

Metrics:
    IoU (Jaccard)
    Dice (F1 over pixels)
    centroid_dist_px
    area_a, area_b, area_diff, area_ratio
    boundary_iou (IoU computed on 1-pixel-thick boundaries only)
    hausdorff_95_px (95-th percentile boundary distance, robust to outliers)

Usage (as library):
    from compare_masks import compare_masks_pair
    metrics = compare_masks_pair(mask_a, mask_b)  # both 2D bool arrays

Usage (as script):
    python compare_masks.py --a path/to/a --b path/to/b --out output_dir
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist
from skimage.morphology import binary_erosion


def _boundary(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask)
    return mask & ~eroded


def compare_masks_pair(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two 2-D binary masks (same shape). Returns metric dict."""
    a = a.astype(bool)
    b = b.astype(bool)

    area_a = int(a.sum())
    area_b = int(b.sum())
    intersect = int((a & b).sum())
    union = int((a | b).sum())

    iou = intersect / union if union > 0 else 0.0
    dice = 2 * intersect / (area_a + area_b) if (area_a + area_b) > 0 else 0.0

    if area_a > 0 and area_b > 0:
        cy_a, cx_a = center_of_mass(a)
        cy_b, cx_b = center_of_mass(b)
        centroid_dist = float(np.hypot(cy_a - cy_b, cx_a - cx_b))
    else:
        centroid_dist = float('nan')

    # Boundary IoU
    bd_a = _boundary(a)
    bd_b = _boundary(b)
    bd_intersect = int((bd_a & bd_b).sum())
    bd_union = int((bd_a | bd_b).sum())
    boundary_iou = bd_intersect / bd_union if bd_union > 0 else 0.0

    # Hausdorff 95: 95th-percentile of boundary-to-boundary distance
    if bd_a.sum() > 0 and bd_b.sum() > 0:
        pts_a = np.argwhere(bd_a)
        pts_b = np.argwhere(bd_b)
        # Subsample for speed if very large boundaries
        if len(pts_a) > 1500:
            pts_a = pts_a[np.random.choice(len(pts_a), 1500, replace=False)]
        if len(pts_b) > 1500:
            pts_b = pts_b[np.random.choice(len(pts_b), 1500, replace=False)]
        d_ab = cdist(pts_a, pts_b).min(axis=1)
        d_ba = cdist(pts_b, pts_a).min(axis=1)
        hausdorff95 = float(max(np.percentile(d_ab, 95), np.percentile(d_ba, 95)))
    else:
        hausdorff95 = float('nan')

    return {
        'area_a': area_a,
        'area_b': area_b,
        'area_diff': abs(area_a - area_b),
        'area_ratio': min(area_a, area_b) / max(area_a, area_b, 1),
        'iou': float(iou),
        'dice': float(dice),
        'centroid_dist_px': centroid_dist,
        'boundary_iou': float(boundary_iou),
        'hausdorff_95_px': hausdorff95,
    }


def compare_stacks(stack_a: np.ndarray, stack_b: np.ndarray, cell_id: str = '') -> list[dict]:
    """Frame-by-frame comparison of two T×H×W stacks. Returns list of dicts."""
    assert stack_a.shape == stack_b.shape, f'shape mismatch: {stack_a.shape} vs {stack_b.shape}'
    out = []
    T = stack_a.shape[0]
    for t in range(T):
        m = compare_masks_pair(stack_a[t], stack_b[t])
        m['cell'] = cell_id
        m['frame'] = t
        out.append(m)
    return out


def _load_mask_stack(path: Path) -> np.ndarray:
    """Load any .tif as boolean mask stack (treats >0 as True)."""
    img = tifffile.imread(str(path))
    return img > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='directory A (paired *_mask.tif files)')
    ap.add_argument('--b', required=True, help='directory B (paired *_mask.tif files)')
    ap.add_argument('--out', required=True, help='output directory for CSVs')
    ap.add_argument('--pattern', default='*_mask.tif', help='glob pattern for matching files')
    args = ap.parse_args()

    dir_a = Path(args.a)
    dir_b = Path(args.b)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    a_files = {p.relative_to(dir_a): p for p in dir_a.rglob(args.pattern)}
    b_files = {p.relative_to(dir_b): p for p in dir_b.rglob(args.pattern)}
    common = sorted(set(a_files.keys()) & set(b_files.keys()))
    print(f'A: {len(a_files)} files, B: {len(b_files)} files, paired: {len(common)}')

    all_records = []
    summaries = []
    for rel in common:
        a = _load_mask_stack(a_files[rel])
        b = _load_mask_stack(b_files[rel])
        cell_id = str(rel).replace('_mask.tif', '')
        records = compare_stacks(a, b, cell_id=cell_id)
        all_records.extend(records)

        valid = [r for r in records if not np.isnan(r['centroid_dist_px'])]
        if valid:
            summaries.append({
                'cell': cell_id,
                'n_frames': len(records),
                'n_valid': len(valid),
                'mean_iou': float(np.mean([r['iou'] for r in valid])),
                'median_iou': float(np.median([r['iou'] for r in valid])),
                'mean_dice': float(np.mean([r['dice'] for r in valid])),
                'mean_centroid_dist_px': float(np.mean([r['centroid_dist_px'] for r in valid])),
                'mean_boundary_iou': float(np.mean([r['boundary_iou'] for r in valid])),
                'mean_hausdorff_95_px': float(np.nanmean([r['hausdorff_95_px'] for r in valid])),
            })
            s = summaries[-1]
            print(f'  {cell_id:<25} mean_IoU={s["mean_iou"]:.3f}  '
                  f'centroid={s["mean_centroid_dist_px"]:.1f}px  '
                  f'HD95={s["mean_hausdorff_95_px"]:.1f}px')

    if all_records:
        with open(out_dir / 'comparison_per_frame.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            writer.writeheader()
            writer.writerows(all_records)
        print(f'\nSaved {out_dir / "comparison_per_frame.csv"} ({len(all_records)} rows)')

    if summaries:
        with open(out_dir / 'comparison_summary.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)
        print(f'Saved {out_dir / "comparison_summary.csv"} ({len(summaries)} rows)')


if __name__ == '__main__':
    main()

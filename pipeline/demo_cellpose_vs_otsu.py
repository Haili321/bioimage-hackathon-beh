"""
Demo of compare_masks.py: Cellpose v3 cell mask vs whole-image Otsu baseline.

For each of 30 cells in Categorised_Data:
  1. Generate per-frame Otsu mask (whole-image threshold + morphology cleanup,
     keep largest connected component)
  2. Compare to v3 Cellpose mask using compare_masks_pair
  3. Output per-frame metrics CSV + per-cell summary CSV
  4. Generate a side-by-side visualization on representative cells

This is a stand-in for "Cellpose vs ground truth". Once Edward / Badeer
provide hand-drawn reference masks, swap the Otsu generator for those.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_objects
from skimage.measure import label

sys.path.insert(0, str(Path(__file__).parent))
from compare_masks import compare_stacks

INPUT_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
CENTERED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
OUT_DIR = CENTERED_DIR / 'mask_comparison'
OUT_DIR.mkdir(exist_ok=True)


def otsu_cell_mask_2d(img: np.ndarray) -> np.ndarray:
    """Whole-image Otsu + morphology cleanup, keep largest connected component."""
    if img.max() <= img.min():
        return np.zeros_like(img, dtype=bool)
    try:
        thresh = threshold_otsu(img)
    except Exception:
        return np.zeros_like(img, dtype=bool)
    binary = img > thresh
    binary = binary_closing(binary, disk(3))
    binary = remove_small_objects(binary, min_size=500)
    if binary.sum() == 0:
        return binary
    lbl = label(binary)
    sizes = np.bincount(lbl.ravel())
    sizes[0] = 0
    largest = int(sizes.argmax())
    return lbl == largest


def main():
    folders = ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']
    all_records = []
    summaries = []

    for folder in folders:
        for raw_path in sorted((INPUT_DIR / folder).glob('*.tif')):
            stem = raw_path.stem
            cell_id = f'{folder}/{stem}'
            mask_path = CENTERED_DIR / folder / f'{stem}_mask.tif'
            if not mask_path.exists():
                continue

            raw = tifffile.imread(raw_path)
            cellpose_mask = tifffile.imread(mask_path) > 0
            assert raw.shape == cellpose_mask.shape, f'{cell_id}: shape mismatch'

            # Generate Otsu masks frame by frame
            T = raw.shape[0]
            otsu_mask = np.zeros_like(cellpose_mask)
            for t in range(T):
                otsu_mask[t] = otsu_cell_mask_2d(raw[t])

            records = compare_stacks(cellpose_mask, otsu_mask, cell_id=cell_id)
            all_records.extend(records)

            # Per-cell summary (skip frames where either mask is empty)
            valid = [r for r in records if r['area_a'] > 0 and r['area_b'] > 0]
            if valid:
                ious = np.array([r['iou'] for r in valid])
                dices = np.array([r['dice'] for r in valid])
                cd = np.array([r['centroid_dist_px'] for r in valid])
                hd95 = np.array([r['hausdorff_95_px'] for r in valid])
                bd_iou = np.array([r['boundary_iou'] for r in valid])
                summaries.append({
                    'cell': cell_id,
                    'condition': folder[:2],
                    'category': folder[2:],
                    'batch': 'low' if folder.endswith('1') else 'high',
                    'T': T,
                    'n_valid_pairs': len(valid),
                    'mean_iou': float(ious.mean()),
                    'median_iou': float(np.median(ious)),
                    'mean_dice': float(dices.mean()),
                    'mean_centroid_dist_px': float(cd.mean()),
                    'mean_boundary_iou': float(bd_iou.mean()),
                    'mean_hausdorff_95_px': float(np.nanmean(hd95)),
                })
                s = summaries[-1]
                print(f'  {cell_id:<25} N={len(valid):>3}/{T:<3}  '
                      f'IoU={s["mean_iou"]:.3f}  Dice={s["mean_dice"]:.3f}  '
                      f'centroid_dist={s["mean_centroid_dist_px"]:.1f}px  '
                      f'HD95={s["mean_hausdorff_95_px"]:.1f}px')

    # Write CSVs
    if all_records:
        with open(OUT_DIR / 'cellpose_vs_otsu_per_frame.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(all_records[0].keys()))
            w.writeheader(); w.writerows(all_records)
    if summaries:
        with open(OUT_DIR / 'cellpose_vs_otsu_summary.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader(); w.writerows(summaries)
    print(f'\nSaved CSVs to {OUT_DIR}')

    # Per-condition aggregate
    print('\n=== Per-condition mean IoU (Cellpose vs Otsu) ===')
    for cond in ['wt', 'ko', 'ki']:
        cond_rows = [s for s in summaries if s['condition'] == cond]
        if cond_rows:
            ious = np.array([s['mean_iou'] for s in cond_rows])
            print(f'  {cond.upper()} (N={len(cond_rows)}): IoU={ious.mean():.3f} ± {ious.std():.3f}')

    # ============================================================
    # Visualization: 6 cells, one per folder, mid-frame, side-by-side
    # ============================================================
    samples = [
        ('ki1/3.tif', 'KI low'),
        ('ki2/19.tif', 'KI high'),
        ('ko1/22.tif', 'KO low'),
        ('ko2/ko11.tif', 'KO high'),
        ('wt1/4.tif', 'WT low'),
        ('wt2/wt4.tif', 'WT high'),
    ]
    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 18))
    for row, (rel, label) in enumerate(samples):
        folder = rel.split('/')[0]
        stem = Path(rel).stem
        raw = tifffile.imread(INPUT_DIR / rel)
        cellpose = tifffile.imread(CENTERED_DIR / folder / f'{stem}_mask.tif') > 0
        T = raw.shape[0]; t = T // 2
        img = raw[t]
        cp_m = cellpose[t]
        otsu_m = otsu_cell_mask_2d(img)
        comp = compare_stacks(cellpose[t:t+1], np.array([otsu_m]), cell_id=rel)[0]
        vmax = np.percentile(img, 99.5)

        # Col 0: raw
        axes[row, 0].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 0].set_title(f'{label}: {rel}\nframe {t}/{T}', fontsize=9)
        axes[row, 0].axis('off')

        # Col 1: Cellpose mask
        axes[row, 1].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 1].imshow(np.ma.masked_where(~cp_m, cp_m.astype(float)), cmap='Greens', alpha=0.5)
        axes[row, 1].set_title(f'Cellpose mask\narea = {int(cp_m.sum())} px', fontsize=9, color='green')
        axes[row, 1].axis('off')

        # Col 2: Otsu mask
        axes[row, 2].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 2].imshow(np.ma.masked_where(~otsu_m, otsu_m.astype(float)), cmap='Oranges', alpha=0.5)
        axes[row, 2].set_title(f'Otsu mask (baseline)\narea = {int(otsu_m.sum())} px', fontsize=9, color='darkorange')
        axes[row, 2].axis('off')

        # Col 3: Difference
        # Green = both (TP), Red = Cellpose only (FN by Otsu), Blue = Otsu only (FP by Otsu)
        only_cp = cp_m & ~otsu_m
        only_otsu = ~cp_m & otsu_m
        both = cp_m & otsu_m
        rgba = np.zeros((*img.shape, 4))
        rgba[..., 0] = only_cp.astype(float) * 0.95
        rgba[..., 1] = both.astype(float) * 0.75
        rgba[..., 2] = only_otsu.astype(float) * 0.95
        rgba[..., 3] = ((only_cp | both | only_otsu).astype(float)) * 0.55
        axes[row, 3].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 3].imshow(rgba)
        axes[row, 3].set_title(f'Difference\nIoU={comp["iou"]:.2f}  Dice={comp["dice"]:.2f}  HD95={comp["hausdorff_95_px"]:.1f}px',
                                fontsize=9)
        axes[row, 3].axis('off')

    fig.suptitle('Mask comparison demo: Cellpose v3 vs whole-image Otsu (one cell per folder, mid-frame)\n'
                 'Difference column: green = both agree, red = Cellpose only, blue = Otsu only',
                 fontsize=11, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = INPUT_DIR / 'cellpose_vs_otsu_demo.png'
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f'Saved {out_png}')


if __name__ == '__main__':
    main()

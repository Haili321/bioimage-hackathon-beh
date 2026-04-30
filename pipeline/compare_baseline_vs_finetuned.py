"""
Compare cyto3 baseline (v3 pipeline output) against fine-tuned cyto3 output
on the 30-cell Categorised_Data dataset.

Reports per-cell and overall:
  - Empty mask rate before vs after
  - Mean IoU between baseline and finetuned masks (do they agree?)
  - Mean mask area before vs after
  - Per-condition aggregate

Generates a side-by-side visualization on representative cells.
"""
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE_DIR  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
FINETUNED_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
INPUT_DIR     = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
OUT_DIR       = FINETUNED_DIR / 'comparison_vs_baseline'
OUT_DIR.mkdir(exist_ok=True)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter / union) if union > 0 else 0.0


def main():
    folders = ['ki1', 'ki2', 'ko1', 'ko2', 'wt1', 'wt2']
    summary = []
    total_baseline_empty = 0
    total_finetuned_empty = 0
    total_frames = 0

    for folder in folders:
        for raw_path in sorted((INPUT_DIR / folder).glob('*.tif')):
            stem = raw_path.stem
            cell_id = f'{folder}/{stem}'
            base_mask_path = BASELINE_DIR / folder / f'{stem}_mask.tif'
            fine_mask_path = FINETUNED_DIR / folder / f'{stem}_mask.tif'

            if not base_mask_path.exists() or not fine_mask_path.exists():
                print(f'[skip] {cell_id} missing one of the masks')
                continue

            base_mask = tifffile.imread(base_mask_path) > 0
            fine_mask = tifffile.imread(fine_mask_path) > 0
            assert base_mask.shape == fine_mask.shape
            T = base_mask.shape[0]
            total_frames += T

            # Per-frame metrics
            base_areas = base_mask.sum(axis=(1, 2))
            fine_areas = fine_mask.sum(axis=(1, 2))
            base_empty = int((base_areas == 0).sum())
            fine_empty = int((fine_areas == 0).sum())
            total_baseline_empty += base_empty
            total_finetuned_empty += fine_empty

            ious = []
            for t in range(T):
                if base_areas[t] > 0 and fine_areas[t] > 0:
                    ious.append(iou(base_mask[t], fine_mask[t]))
            mean_iou = float(np.mean(ious)) if ious else 0.0

            cond = folder[:2]
            cat = folder[2:]
            batch = 'low' if cat == '1' else 'high'

            summary.append({
                'cell': cell_id,
                'condition': cond,
                'category': cat,
                'batch': batch,
                'T': T,
                'baseline_empty': base_empty,
                'baseline_empty_pct': base_empty / T * 100,
                'finetuned_empty': fine_empty,
                'finetuned_empty_pct': fine_empty / T * 100,
                'delta_empty_pct': (fine_empty - base_empty) / T * 100,
                'mean_iou_base_vs_fine': mean_iou,
                'baseline_mean_area': float(base_areas[base_areas > 0].mean()) if (base_areas > 0).any() else 0,
                'finetuned_mean_area': float(fine_areas[fine_areas > 0].mean()) if (fine_areas > 0).any() else 0,
            })

            print(f'  {cell_id:<25} '
                  f'empty: base {base_empty}/{T} ({base_empty/T*100:.1f}%) → '
                  f'fine {fine_empty}/{T} ({fine_empty/T*100:.1f}%)  '
                  f'IoU(base,fine)={mean_iou:.3f}')

    # Save summary CSV
    with open(OUT_DIR / 'baseline_vs_finetuned_summary.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f'\nSaved {OUT_DIR / "baseline_vs_finetuned_summary.csv"}')

    # Overall stats
    print('\n=== Overall ===')
    print(f'Total frames:       {total_frames}')
    print(f'Baseline empty:     {total_baseline_empty} ({total_baseline_empty/total_frames*100:.2f}%)')
    print(f'Finetuned empty:    {total_finetuned_empty} ({total_finetuned_empty/total_frames*100:.2f}%)')
    print(f'Delta:              {total_finetuned_empty - total_baseline_empty:+d} '
          f'({(total_finetuned_empty - total_baseline_empty)/total_frames*100:+.2f}%)')

    print('\n=== Per condition ===')
    for cond in ['wt', 'ko', 'ki']:
        cond_rows = [s for s in summary if s['condition'] == cond]
        if not cond_rows: continue
        be = sum(s['baseline_empty'] for s in cond_rows)
        fe = sum(s['finetuned_empty'] for s in cond_rows)
        tT = sum(s['T'] for s in cond_rows)
        ious = np.array([s['mean_iou_base_vs_fine'] for s in cond_rows])
        print(f'  {cond.upper()} (N={len(cond_rows)}): '
              f'empty {be/tT*100:.2f}% → {fe/tT*100:.2f}%  '
              f'IoU(base, fine) = {ious.mean():.3f} ± {ious.std():.3f}')

    # ============================================================
    # Visualization
    # ============================================================
    samples = [
        ('ki1/3.tif',   'ki1/3 (KI low)'),
        ('ko1/22.tif',  'ko1/22 (KO low)'),
        ('wt1/4.tif',   'wt1/4 (WT low)'),
        ('ki2/19.tif',  'ki2/19 (KI high)'),
        ('ko2/ko11.tif','ko2/ko11 (KO high)'),
        ('wt2/wt4.tif', 'wt2/wt4 (WT high)'),
    ]
    fig, axes = plt.subplots(len(samples), 4, figsize=(15, 18))
    for row, (rel, label) in enumerate(samples):
        folder = rel.split('/')[0]
        stem = Path(rel).stem
        raw = tifffile.imread(INPUT_DIR / rel)
        base = tifffile.imread(BASELINE_DIR / folder / f'{stem}_mask.tif') > 0
        fine = tifffile.imread(FINETUNED_DIR / folder / f'{stem}_mask.tif') > 0
        T = raw.shape[0]; t = T // 2
        img = raw[t]; bm = base[t]; fm = fine[t]
        comp_iou = iou(bm, fm)
        vmax = np.percentile(img, 99.5)

        axes[row, 0].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 0].set_title(f'{label}\nframe {t}/{T}', fontsize=9)
        axes[row, 0].axis('off')

        axes[row, 1].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 1].imshow(np.ma.masked_where(~bm, bm.astype(float)), cmap='Greens', alpha=0.5)
        axes[row, 1].set_title(f'cyto3 baseline\narea={int(bm.sum())} px', fontsize=9, color='green')
        axes[row, 1].axis('off')

        axes[row, 2].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 2].imshow(np.ma.masked_where(~fm, fm.astype(float)), cmap='Purples', alpha=0.5)
        axes[row, 2].set_title(f'fine-tuned\narea={int(fm.sum())} px', fontsize=9, color='purple')
        axes[row, 2].axis('off')

        only_b = bm & ~fm
        only_f = ~bm & fm
        both = bm & fm
        rgba = np.zeros((*img.shape, 4))
        rgba[..., 0] = only_b.astype(float) * 0.95
        rgba[..., 1] = both.astype(float) * 0.75
        rgba[..., 2] = only_f.astype(float) * 0.95
        rgba[..., 3] = ((only_b | both | only_f).astype(float)) * 0.55
        axes[row, 3].imshow(img, cmap='gray', vmax=vmax)
        axes[row, 3].imshow(rgba)
        axes[row, 3].set_title(f'difference\nIoU={comp_iou:.2f}', fontsize=9)
        axes[row, 3].axis('off')

    fig.suptitle('cyto3 baseline vs fine-tuned cyto3 (one cell per folder, mid-frame)\n'
                 'Difference column: green = both, red = baseline only, blue = fine-tuned only',
                 fontsize=11, y=0.998)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = INPUT_DIR / 'baseline_vs_finetuned_demo.png'
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    print(f'\nSaved {out_png}')


if __name__ == '__main__':
    main()

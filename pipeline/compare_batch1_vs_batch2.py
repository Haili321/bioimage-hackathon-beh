"""
Generalisation test: compare findings on the original 30 cells (batch 1)
against the 2nd-upload 30 cells (batch 2), using the SAME fine-tuned
Cellpose model.

Outputs:
  comparison_batch1_vs_batch2.csv   -- per-condition summary (both batches)
  comparison_batch1_vs_batch2.png   -- migration + lamellipodia side-by-side
  empty_rate_batch1_vs_batch2.csv   -- per-cell segmentation quality
  cohens_d_batch1_vs_batch2.csv     -- effect-size comparison

Reads:
  Categorised_Data_finetuned/trajectories/migration_metrics.csv      (batch 1)
  Categorised_Data_finetuned/lamellipodia/per_cell_summary.csv       (batch 1)
  Categorised_Data_2nd_finetuned/trajectories/migration_metrics.csv  (batch 2)
  Categorised_Data_2nd_finetuned/lamellipodia/per_cell_summary.csv   (batch 2)
  Categorised_Data_finetuned/centering_manifest.json                 (batch 1)
  Categorised_Data_2nd_finetuned/centering_manifest.json             (batch 2)
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

B1 = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
B2 = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_finetuned')
OUT_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data/comparison_batch1_vs_batch2')
OUT_DIR.mkdir(exist_ok=True, parents=True)

CONDS = ['wt', 'ko', 'ki']
COLORS = {'wt': '#1f77b4', 'ko': '#ff7f0e', 'ki': '#d62728'}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def cohens_d(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / max(np.sqrt(sp2), 1e-9))


def empty_rate(manifest_path):
    with open(manifest_path) as f:
        m = json.load(f)
    out = []
    for rel, info in m.items():
        T = info['shape'][0]
        n_e = info['n_empty_masks']
        out.append({
            'cell': rel.replace('.tif', ''),
            'folder': rel.split('/')[0],
            'condition': rel.split('/')[0][:2],
            'T': T,
            'n_empty': n_e,
            'empty_rate': n_e / T if T > 0 else 0.0,
        })
    return out


def per_cond_stats(rows, key):
    out = {}
    for c in CONDS:
        vals = np.array([float(r[key]) for r in rows if r['condition'] == c])
        out[c] = (float(vals.mean()) if len(vals) else np.nan,
                  float(vals.std()) if len(vals) else np.nan,
                  vals)
    return out


def main():
    mig1 = read_csv(B1 / 'trajectories' / 'migration_metrics.csv')
    mig2 = read_csv(B2 / 'trajectories' / 'migration_metrics.csv')
    lam1 = read_csv(B1 / 'lamellipodia' / 'per_cell_summary.csv')
    lam2 = read_csv(B2 / 'lamellipodia' / 'per_cell_summary.csv')
    e1 = empty_rate(B1 / 'centering_manifest.json')
    e2 = empty_rate(B2 / 'centering_manifest.json')

    # Empty rate
    print('=== Segmentation quality (empty mask rate) ===')
    total1 = sum(r['T'] for r in e1); empty1 = sum(r['n_empty'] for r in e1)
    total2 = sum(r['T'] for r in e2); empty2 = sum(r['n_empty'] for r in e2)
    print(f'  batch 1 ({len(e1)} cells): {empty1}/{total1} empty = {100*empty1/total1:.2f}%')
    print(f'  batch 2 ({len(e2)} cells): {empty2}/{total2} empty = {100*empty2/total2:.2f}%')

    with open(OUT_DIR / 'empty_rate_batch1_vs_batch2.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch', 'cell', 'folder', 'condition', 'T', 'n_empty', 'empty_rate'])
        for r in e1: w.writerow(['1', r['cell'], r['folder'], r['condition'], r['T'], r['n_empty'], r['empty_rate']])
        for r in e2: w.writerow(['2', r['cell'], r['folder'], r['condition'], r['T'], r['n_empty'], r['empty_rate']])

    # Migration: mean speed
    print('\n=== Migration (mean speed um/min) ===')
    s1 = per_cond_stats(mig1, 'mean_speed_um_per_min')
    s2 = per_cond_stats(mig2, 'mean_speed_um_per_min')
    print(f'  {"":4} {"batch 1":>16} {"batch 2":>16}')
    for c in CONDS:
        m1, sd1, _ = s1[c]; m2, sd2, _ = s2[c]
        print(f'  {c.upper():4} {m1:>5.2f} ± {sd1:>4.2f}    {m2:>5.2f} ± {sd2:>4.2f}')

    # Lamellipodia: lam/cyto ratio
    print('\n=== Lamellipodia (lam/cyto intensity ratio) ===')
    l1 = per_cond_stats(lam1, 'mean_lam_to_cyto_I_ratio')
    l2 = per_cond_stats(lam2, 'mean_lam_to_cyto_I_ratio')
    print(f'  {"":4} {"batch 1":>16} {"batch 2":>16}')
    for c in CONDS:
        m1, sd1, _ = l1[c]; m2, sd2, _ = l2[c]
        print(f'  {c.upper():4} {m1:>5.3f} ± {sd1:>4.3f}    {m2:>5.3f} ± {sd2:>4.3f}')

    # Effect sizes
    print("\n=== Cohen's d (KI vs WT, KI vs KO) ===")
    rows_d = []
    for label, statset in [('migration_speed', (s1, s2)), ('lam_cyto_ratio', (l1, l2))]:
        for batch_idx, st in enumerate(statset, 1):
            d_ki_wt = cohens_d(st['ki'][2], st['wt'][2])
            d_ki_ko = cohens_d(st['ki'][2], st['ko'][2])
            print(f'  {label:>20} batch {batch_idx}: KI vs WT d={d_ki_wt:>+.2f}  KI vs KO d={d_ki_ko:>+.2f}')
            rows_d.append({'metric': label, 'batch': batch_idx,
                           'cohens_d_ki_vs_wt': d_ki_wt, 'cohens_d_ki_vs_ko': d_ki_ko})
    with open(OUT_DIR / 'cohens_d_batch1_vs_batch2.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_d[0].keys())); w.writeheader(); w.writerows(rows_d)

    # Per-condition summary csv
    sum_rows = []
    for batch_idx, (sm, sl) in enumerate([(s1, l1), (s2, l2)], 1):
        for c in CONDS:
            sum_rows.append({
                'batch': batch_idx,
                'condition': c,
                'n_cells': len(sm[c][2]),
                'mean_speed_um_per_min': sm[c][0],
                'std_speed': sm[c][1],
                'mean_lam_cyto_ratio': sl[c][0],
                'std_lam_cyto_ratio': sl[c][1],
            })
    with open(OUT_DIR / 'comparison_batch1_vs_batch2.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys())); w.writeheader(); w.writerows(sum_rows)

    # Figure: 2 panels (migration / lam), grouped bars
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    def grouped_bar(ax, stats1, stats2, ylabel, title):
        x = np.arange(len(CONDS))
        w = 0.36
        m1 = [stats1[c][0] for c in CONDS]; sd1 = [stats1[c][1] for c in CONDS]
        m2 = [stats2[c][0] for c in CONDS]; sd2 = [stats2[c][1] for c in CONDS]
        b1 = ax.bar(x - w/2, m1, w, yerr=sd1, label='batch 1 (original 30)',
                    color=[COLORS[c] for c in CONDS], edgecolor='black', linewidth=0.8, capsize=3)
        b2 = ax.bar(x + w/2, m2, w, yerr=sd2, label='batch 2 (fresh 30)',
                    color=[COLORS[c] for c in CONDS], edgecolor='black', linewidth=0.8, capsize=3,
                    hatch='//', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in CONDS])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='best', fontsize=9)

    grouped_bar(axes[0], s1, s2, 'mean speed (μm/min)', 'Migration speed')
    grouped_bar(axes[1], l1, l2, 'lam / cyto intensity ratio', 'Arp2/3 lamellipodia polarisation')
    fig.suptitle('Generalisation: same fine-tuned model on two batches', y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'comparison_batch1_vs_batch2.png', dpi=150, bbox_inches='tight')
    print(f'\n[done] saved figure: {OUT_DIR / "comparison_batch1_vs_batch2.png"}')


if __name__ == '__main__':
    main()

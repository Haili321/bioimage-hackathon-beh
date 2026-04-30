"""
Compare self-trained vs Quimp-trained model results, on both batches.
Reads existing CSVs only; does not modify anything.

Outputs to a new directory:
    Categorised_Data/self_vs_quimp/
        comparison_table.csv
        cohens_d_comparison.csv
        summary.txt
"""
import csv
import json
from pathlib import Path

import numpy as np

PATHS = {
    'self_b1':  Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned'),
    'self_b2':  Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_finetuned'),
    'quimp_b1': Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned'),
    'quimp_b2': Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned'),
}
OUT = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data/self_vs_quimp')
OUT.mkdir(parents=True, exist_ok=True)

CONDS = ['wt', 'ko', 'ki']


def read_csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def per_cond_stats(rows, key):
    out = {}
    for c in CONDS:
        vals = np.array([float(r[key]) for r in rows if r['condition'] == c])
        out[c] = vals
    return out


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float('nan')
    sp2 = ((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2)
    return float((x.mean() - y.mean()) / max(np.sqrt(sp2), 1e-9))


def main():
    rows = {}
    rows_lam = {}

    for tag, base in PATHS.items():
        mig_path = base / 'trajectories' / 'migration_metrics.csv'
        lam_path = base / 'lamellipodia' / 'per_cell_summary.csv'
        if not mig_path.exists() or not lam_path.exists():
            print(f'[skip] {tag}: missing CSVs')
            rows[tag] = None; rows_lam[tag] = None
            continue
        rows[tag] = read_csv(mig_path)
        rows_lam[tag] = read_csv(lam_path)

    # Empty rates
    print('=== Empty mask rate (segmentation quality) ===')
    for tag, base in PATHS.items():
        manifest_path = base / 'centering_manifest.json'
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            m = json.load(f)
        T = sum(i['shape'][0] for i in m.values())
        E = sum(i['n_empty_masks'] for i in m.values())
        print(f'  {tag:<10}: {E:>4}/{T:<5} = {100*E/T:.3f}%')

    # Migration speed
    print('\n=== Migration speed (mean +/- std, μm/min) ===')
    print(f'  {"":>4} {"self_b1":>14} {"quimp_b1":>14}  {"self_b2":>14} {"quimp_b2":>14}')
    for c in CONDS:
        line = [f'  {c.upper():>4}']
        for tag in ['self_b1', 'quimp_b1', 'self_b2', 'quimp_b2']:
            if rows.get(tag) is None:
                line.append(f'{"":>14}')
                continue
            v = np.array([float(r['mean_speed_um_per_min']) for r in rows[tag] if r['condition'] == c])
            line.append(f'{v.mean():>5.2f} ± {v.std():.2f}'.rjust(14))
        print(' '.join(line))

    # Lam ratio
    print('\n=== Lamellipodia / cytoplasm intensity ratio (mean +/- std) ===')
    print(f'  {"":>4} {"self_b1":>14} {"quimp_b1":>14}  {"self_b2":>14} {"quimp_b2":>14}')
    for c in CONDS:
        line = [f'  {c.upper():>4}']
        for tag in ['self_b1', 'quimp_b1', 'self_b2', 'quimp_b2']:
            if rows_lam.get(tag) is None:
                line.append(f'{"":>14}')
                continue
            v = np.array([float(r['mean_lam_to_cyto_I_ratio']) for r in rows_lam[tag] if r['condition'] == c])
            line.append(f'{v.mean():>5.3f} ± {v.std():.3f}'.rjust(14))
        print(' '.join(line))

    # Cohen's d (KI vs WT, KI vs KO) for both metrics on both batches and both models
    print("\n=== Cohen's d (KI signal strength) ===")
    rows_d = []
    for metric_label, source in [('migration_speed', rows), ('lam_cyto_ratio', rows_lam)]:
        key = 'mean_speed_um_per_min' if metric_label == 'migration_speed' else 'mean_lam_to_cyto_I_ratio'
        for tag in ['self_b1', 'quimp_b1', 'self_b2', 'quimp_b2']:
            if source.get(tag) is None:
                continue
            vals = per_cond_stats(source[tag], key)
            d_ki_wt = cohens_d(vals['ki'], vals['wt'])
            d_ki_ko = cohens_d(vals['ki'], vals['ko'])
            rows_d.append({
                'metric': metric_label, 'tag': tag,
                'cohens_d_ki_vs_wt': d_ki_wt,
                'cohens_d_ki_vs_ko': d_ki_ko,
            })
            print(f'  {metric_label:<18}  {tag:<10}  KI vs WT d={d_ki_wt:+.2f}  KI vs KO d={d_ki_ko:+.2f}')

    if rows_d:
        with open(OUT / 'cohens_d_comparison.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_d[0].keys())); w.writeheader(); w.writerows(rows_d)
        print(f'\n[saved] {OUT / "cohens_d_comparison.csv"}')


if __name__ == '__main__':
    main()

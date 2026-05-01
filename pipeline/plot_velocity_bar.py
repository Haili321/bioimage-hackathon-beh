"""
Slide-ready bar chart of mean migration speed by condition (WT/KO/KI),
model (self vs Quimp) and batch (B1 vs B2).

Per-cell dots are scatter-overlayed on top of the bars to show within-condition
spread directly (no need to read off error bars alone).

Output (chmod 644):
  ~/public_html/hackathon-beh/velocity_bar_chart.png
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PATHS = {
    'self_b1':  Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned'),
    'self_b2':  Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_finetuned'),
    'quimp_b1': Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned'),
    'quimp_b2': Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned'),
}
OUT = Path('/dcs/pg25/u1898019/public_html/hackathon-beh/velocity_bar_chart.png')

CONDS = ['wt', 'ko', 'ki']
COND_LABEL = {'wt': 'WT', 'ko': 'KO', 'ki': 'KI'}
TAGS = ['self_b1', 'quimp_b1', 'self_b2', 'quimp_b2']
TAG_LABEL = {
    'self_b1':  'self · B1',
    'self_b2':  'self · B2',
    'quimp_b1': 'Quimp · B1',
    'quimp_b2': 'Quimp · B2',
}
TAG_COLOR = {
    'self_b1':  '#1f4e8c',
    'self_b2':  '#5fa8e0',
    'quimp_b1': '#a02020',
    'quimp_b2': '#e07060',
}


def load_speeds(base):
    p = base / 'trajectories' / 'migration_metrics.csv'
    rows = list(csv.DictReader(open(p)))
    out = {}
    for c in CONDS:
        out[c] = np.array([float(r['mean_speed_um_per_min'])
                           for r in rows if r['condition'] == c])
    return out


speeds = {tag: load_speeds(base) for tag, base in PATHS.items()}

fig, ax = plt.subplots(figsize=(12, 5.5))
plt.rcParams.update({'font.size': 11})

n_groups = len(CONDS)
n_bars = len(TAGS)
bar_w = 0.18
x = np.arange(n_groups)

for i, tag in enumerate(TAGS):
    means = [speeds[tag][c].mean() for c in CONDS]
    stds  = [speeds[tag][c].std()  for c in CONDS]
    bar_x = x + (i - (n_bars - 1) / 2) * bar_w
    ax.bar(bar_x, means, bar_w,
           yerr=stds, capsize=3,
           label=TAG_LABEL[tag], color=TAG_COLOR[tag],
           edgecolor='black', linewidth=0.6, alpha=0.85, zorder=2,
           error_kw={'elinewidth': 1, 'ecolor': '#444'})

    # Per-cell dots
    for cj, c in enumerate(CONDS):
        vals = speeds[tag][c]
        jitter = (np.random.RandomState(cj * 10 + i).rand(len(vals)) - 0.5) * (bar_w * 0.7)
        dot_x = np.full_like(vals, bar_x[cj]) + jitter
        ax.scatter(dot_x, vals, s=14, color='black', alpha=0.55, zorder=4,
                   edgecolors='white', linewidth=0.4)

    # Annotate mean value
    for bx, m in zip(bar_x, means):
        ax.text(bx, m + 0.05, f'{m:.2f}', ha='center', va='bottom',
                fontsize=8, color='#222', zorder=5)

ax.set_xticks(x)
ax.set_xticklabels([COND_LABEL[c] for c in CONDS], fontsize=14, fontweight='bold')
ax.set_ylabel('Mean migration speed (μm/min)', fontsize=12)

YMAX = 2.5
# Count outliers above YMAX
outliers = [(tag, c, v) for tag in TAGS for c in CONDS for v in speeds[tag][c] if v > YMAX]
ax.set_ylim(0, YMAX)
if outliers:
    ax.text(0.99, 0.97,
            f'{len(outliers)} per-cell outlier(s) > {YMAX} not shown',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=8.5, color='#a02020',
            bbox=dict(facecolor='white', edgecolor='#e2e8f0', boxstyle='round,pad=0.3'))
ax.grid(axis='y', alpha=0.25, zorder=0)
ax.set_axisbelow(True)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4,
          fontsize=11, frameon=False, columnspacing=2.0, handletextpad=0.6)

ax.set_title('Cell migration speed by genotype, both models, both batches',
             fontsize=13, color='#1a202c', pad=42)

# Footer
fig.text(0.5, -0.01,
         'Bar height = mean across cells. Error bars = ±1 std. Dots = individual cells (N=10 per condition per batch). '
         'KI is the slowest condition under all four model × batch combinations.',
         ha='center', fontsize=9, color='#4a5568')

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
import os; os.chmod(OUT, 0o644)
print(f'[saved] {OUT}')

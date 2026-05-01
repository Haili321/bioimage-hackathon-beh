"""
Slide-ready figure: Cohen's d for the two KI findings under all four
combinations of model (self-trained / Quimp-trained) and batch (B1 / B2).

Reads:
  Categorised_Data/self_vs_quimp/cohens_d_comparison.csv

Output (16:9 landscape, slide-friendly):
  ~/public_html/hackathon-beh/cohens_d_4quadrant.png   (chmod 644)

Layout: 2 panels side-by-side (migration / lamellipodia). Each panel has
two contrasts (KI vs WT, KI vs KO). For each contrast, four bars
representing the four model x batch combinations. Reference horizontal
lines at the standard Cohen's d magnitude bands.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INP = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data/self_vs_quimp/cohens_d_comparison.csv')
OUT = Path('/dcs/pg25/u1898019/public_html/hackathon-beh/cohens_d_4quadrant.png')

rows = list(csv.DictReader(open(INP)))

# Re-shape: dict keyed by (metric, tag) -> {KI_vs_WT, KI_vs_KO}
data = {}
for r in rows:
    key = (r['metric'], r['tag'])
    data[key] = {
        'KI vs WT': float(r['cohens_d_ki_vs_wt']),
        'KI vs KO': float(r['cohens_d_ki_vs_ko']),
    }

# Order
TAGS = ['self_b1', 'quimp_b1', 'self_b2', 'quimp_b2']
TAG_LABEL = {
    'self_b1':  'self · B1',
    'self_b2':  'self · B2',
    'quimp_b1': 'Quimp · B1',
    'quimp_b2': 'Quimp · B2',
}
TAG_COLOR = {
    'self_b1':  '#1f4e8c',   # dark blue
    'self_b2':  '#5fa8e0',   # light blue
    'quimp_b1': '#a02020',   # dark red
    'quimp_b2': '#e07060',   # light red
}
CONTRASTS = ['KI vs WT', 'KI vs KO']

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
plt.rcParams.update({'font.size': 11})

PANELS = [
    ('migration_speed', axes[0], "Migration speed: |Cohen's d|", 'KI cells migrate slower'),
    ('lam_cyto_ratio',  axes[1], "Lam / cyto ratio: Cohen's d",   'KI cells more polarised'),
]

ref_lines = [(0.2, 'small'), (0.5, 'medium'), (0.8, 'large'), (1.0, 'very large')]

for metric, ax, ylabel, headline in PANELS:
    n_groups = len(CONTRASTS)
    n_bars   = len(TAGS)
    bar_w = 0.18
    x = np.arange(n_groups)

    # Whether the metric reads natively negative (migration) — flip sign so bars point up = stronger KI difference
    is_neg = (metric == 'migration_speed')

    for i, tag in enumerate(TAGS):
        vals = [data[(metric, tag)][c] for c in CONTRASTS]
        plot_vals = [-v if is_neg else v for v in vals]
        bar_x = x + (i - (n_bars - 1) / 2) * bar_w
        bars = ax.bar(bar_x, plot_vals, bar_w,
                      label=TAG_LABEL[tag], color=TAG_COLOR[tag],
                      edgecolor='black', linewidth=0.6, zorder=3)
        # Annotate value on top
        for bx, bv, raw in zip(bar_x, plot_vals, vals):
            sign = '−' if raw < 0 else '+'
            ax.text(bx, bv + 0.02, f'{sign}{abs(raw):.2f}',
                    ha='center', va='bottom', fontsize=8, color='#222', zorder=5)

    # Reference horizontal lines
    for d_val, lbl in ref_lines:
        ax.axhline(d_val, color='#999', linestyle='--', linewidth=0.7, zorder=1)
        ax.text(n_groups - 0.49, d_val + 0.015, lbl, fontsize=8, color='#777',
                ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(CONTRASTS, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_title(headline, fontsize=12, color='#2d3748', pad=10)
    ax.set_ylim(0, 2.0)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

# Single shared legend at the top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=11,
           bbox_to_anchor=(0.5, 1.05), frameon=False,
           columnspacing=2.0, handletextpad=0.6)

fig.suptitle("KI signal strength across 4 settings (2 models × 2 batches): direction holds everywhere",
             fontsize=13, y=1.12, color='#1a202c')

# A small footer note
fig.text(0.5, -0.02,
         'Bars show the magnitude of Cohen\'s d. Migration values are natively negative (KI slower); lam values are positive (KI more polarised). '
         'All 16 bars point in the predicted direction; lam effect size is essentially identical between self-trained and Quimp-trained models.',
         ha='center', fontsize=9, color='#4a5568')

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='white')
import os; os.chmod(OUT, 0o644)
print(f'[saved] {OUT}')

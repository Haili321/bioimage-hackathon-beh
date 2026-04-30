"""
Visual inspection of Quimp vs Cellpose mask mismatch on selected cells.
For each cell, plot 3 representative frames (early/mid/late) showing:
  raw centered image, Quimp mask, Cellpose mask, and a difference overlay.

This is read-only on existing files. Output goes to a new directory.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile

QUIMP = Path('/dcs/pg25/u1898019/Desktop/Quimp_seg4CellPose_Retrain')
OURS  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
OUT   = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data/quimp_inspect')
OUT.mkdir(parents=True, exist_ok=True)

CELLS = [
    # picked: a "high IoU when same cell" + a "low overlap, probably same cell larger" + a "different cell"
    ('ko2', 'ko1_snakemask',          'ko1',  'high IoU 0.88 (best agreement)'),
    ('ki1', '23_centered_snakemask',  '23',   'low IoU 0.04 but 90% overlap (Quimp much larger)'),
    ('ko2', 'ko11_centered_snakemask','ko11', 'low IoU 0.12 + 41% overlap (likely different cells)'),
    ('wt1', '33_centered_snakemask',  '33',   'low IoU 0.19 + 100% overlap (multi-component disagreement)'),
    ('ki1', '12_centered_snakemask',  '12',   'mid IoU 0.42 + 100% overlap (same cell, area diff 1.48x)'),
]


def load(folder, qstem, ostem):
    Q  = tifffile.imread(QUIMP / folder / f'{qstem}.tif') > 0
    O  = tifffile.imread(OURS  / folder / f'{ostem}_mask.tif') > 0
    Cf = tifffile.imread(OURS  / folder / f'{ostem}_centered.tif')
    return Q, O, Cf


def picks(T):
    if T <= 3:
        return list(range(T))
    return [T // 6, T // 2, 5 * T // 6]


fig, axes = plt.subplots(len(CELLS), 4, figsize=(14, 3.2 * len(CELLS)))
if len(CELLS) == 1:
    axes = axes.reshape(1, -1)

for r, (folder, qstem, ostem, label) in enumerate(CELLS):
    Q, O, C = load(folder, qstem, ostem)
    T = Q.shape[0]
    t = picks(T)[1]   # middle frame
    raw = C[t]
    q   = Q[t]; o = O[t]

    # Centroid markers
    def centroid(m):
        if m.sum() == 0: return None
        ys, xs = np.where(m)
        return ys.mean(), xs.mean()
    qc = centroid(q); oc = centroid(o)

    # Normalise raw for display
    p1, p99 = np.percentile(raw, (1, 99))
    raw_n = np.clip((raw.astype(float) - p1) / max(p99 - p1, 1), 0, 1)

    axes[r, 0].imshow(raw_n, cmap='gray'); axes[r, 0].set_title(f'{folder}/{ostem}  frame {t}\n{label}', fontsize=9)
    axes[r, 1].imshow(raw_n, cmap='gray'); axes[r, 1].imshow(q, cmap='Reds', alpha=0.45)
    if qc: axes[r, 1].plot(qc[1], qc[0], 'r+', markersize=14, mew=2)
    axes[r, 1].set_title(f'Quimp mask  area={q.sum()}', fontsize=9)
    axes[r, 2].imshow(raw_n, cmap='gray'); axes[r, 2].imshow(o, cmap='Blues', alpha=0.45)
    if oc: axes[r, 2].plot(oc[1], oc[0], 'b+', markersize=14, mew=2)
    axes[r, 2].set_title(f'Cellpose mask  area={o.sum()}', fontsize=9)

    # Difference overlay: green = both, red = Quimp only, blue = Cellpose only
    overlay = np.zeros((*q.shape, 3))
    overlay[..., 0] = (q & ~o).astype(float)   # red: Quimp only
    overlay[..., 1] = (q & o).astype(float)    # green: both
    overlay[..., 2] = (~q & o).astype(float)   # blue: Cellpose only
    axes[r, 3].imshow(raw_n, cmap='gray', alpha=0.6); axes[r, 3].imshow(overlay, alpha=0.6)
    iou = (q & o).sum() / max((q | o).sum(), 1)
    cdist = float('nan') if not (qc and oc) else np.sqrt((qc[0]-oc[0])**2 + (qc[1]-oc[1])**2)
    axes[r, 3].set_title(f'overlay   IoU={iou:.3f}  centroid_dist={cdist:.0f}px', fontsize=9)

    for c in range(4):
        axes[r, c].axis('off')

fig.suptitle('Quimp (red +) vs Cellpose fine-tuned (blue +) — middle frame of each cell\n'
             'Overlay: green=both, red=Quimp only, blue=Cellpose only',
             y=1.0, fontsize=11)
fig.tight_layout()
out_path = OUT / 'mismatch_inspection.png'
fig.savefig(out_path, dpi=130, bbox_inches='tight')
print(f'[saved] {out_path}')

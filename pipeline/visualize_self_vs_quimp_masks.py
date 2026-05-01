"""
Side-by-side mask visualisation: self-trained vs Quimp-trained Cellpose
on the same cells / same frames. Read-only on existing files.

Output:
    ~/public_html/hackathon-beh/self_vs_quimp_masks.png   (chmod 644)

Picks a representative mid-frame for each of 6 cells across all 3 genotypes
and both batches. Each row: raw | self mask | quimp mask | overlay.
"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tifffile

SELF_B1 = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned')
SELF_B2 = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_finetuned')
QMP_B1  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned')
QMP_B2  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned')
RAW_B1  = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')          # raw .tif input
RAW_B2  = Path('/dcs/pg25/u1898019/Desktop/2ndUpload')                 # raw .tif input
OUT     = Path('/dcs/pg25/u1898019/public_html/hackathon-beh/self_vs_quimp_masks.png')

# Cell selection: cover all 6 conditions, mix B1 + B2, mix easy + hard
CELLS = [
    # (folder, stem, batch, label)
    ('wt1', '4',     'B1', 'WT (B1, training cell)'),
    ('wt2', 'wt15',  'B2', 'WT (B2, fresh cell)'),
    ('ko1', '8',     'B1', 'KO (B1, stubborn dim cell)'),
    ('ko2', 'ko30',  'B1', 'KO (B1, biologist reference exists)'),
    ('ki1', '7',     'B1', 'KI (B1, training cell)'),
    ('ki2', '30',    'B2', 'KI (B2, fresh cell)'),
]


def load(folder, stem, batch):
    """Loads RAW input + both masks (all in raw frame so they overlay correctly)."""
    if batch == 'B1':
        sd, qd, rd = SELF_B1, QMP_B1, RAW_B1
    else:
        sd, qd, rd = SELF_B2, QMP_B2, RAW_B2
    raw    = tifffile.imread(rd / folder / f'{stem}.tif')
    self_m = tifffile.imread(sd / folder / f'{stem}_mask.tif') > 0
    qmp_m  = tifffile.imread(qd / folder / f'{stem}_mask.tif') > 0
    return raw, self_m, qmp_m


def normalize(img):
    p1, p99 = np.percentile(img, (1, 99))
    return np.clip((img.astype(float) - p1) / max(p99 - p1, 1), 0, 1)


fig, axes = plt.subplots(len(CELLS), 4, figsize=(13, 3.0 * len(CELLS)))
if len(CELLS) == 1:
    axes = axes.reshape(1, -1)

for r, (folder, stem, batch, label) in enumerate(CELLS):
    raw_stack, S, Q = load(folder, stem, batch)
    T = raw_stack.shape[0]
    t = T // 2  # mid frame
    raw = normalize(raw_stack[t])
    s   = S[t]
    q   = Q[t]

    iou = (s & q).sum() / max((s | q).sum(), 1)
    s_area = s.sum(); q_area = q.sum()
    area_ratio = q_area / max(s_area, 1)

    # Col 0: raw centered
    axes[r, 0].imshow(raw, cmap='gray')
    axes[r, 0].set_title(f'{folder}/{stem}  ({batch})\n{label}\nframe {t}/{T}', fontsize=9)

    # Col 1: self-trained mask overlay
    axes[r, 1].imshow(raw, cmap='gray')
    axes[r, 1].imshow(np.where(s, 1.0, np.nan), cmap='Blues', alpha=0.55, vmin=0, vmax=1)
    axes[r, 1].set_title(f'self-trained\narea = {s_area} px', fontsize=9)

    # Col 2: Quimp-trained mask overlay
    axes[r, 2].imshow(raw, cmap='gray')
    axes[r, 2].imshow(np.where(q, 1.0, np.nan), cmap='Reds', alpha=0.55, vmin=0, vmax=1)
    axes[r, 2].set_title(f'Quimp-trained\narea = {q_area} px  (q/s ratio {area_ratio:.2f})', fontsize=9)

    # Col 3: difference overlay
    overlay = np.zeros((*s.shape, 3))
    overlay[..., 0] = (q & ~s).astype(float)   # red: Quimp only
    overlay[..., 1] = (s & q).astype(float)    # green: both
    overlay[..., 2] = (s & ~q).astype(float)   # blue: self only
    # Need RGBA so transparent where no mask
    rgba = np.zeros((*s.shape, 4))
    rgba[..., :3] = overlay
    rgba[..., 3]  = (s | q).astype(float) * 0.65
    axes[r, 3].imshow(raw, cmap='gray')
    axes[r, 3].imshow(rgba)
    axes[r, 3].set_title(f'overlay  IoU = {iou:.3f}\n(green = both, red = quimp only, blue = self only)', fontsize=9)

    for c in range(4):
        axes[r, c].axis('off')

fig.suptitle('Same cells, same frames, two models: self-trained Cellpose (blue) vs Quimp-trained Cellpose (red)',
             y=1.0, fontsize=11)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130, bbox_inches='tight')
import os; os.chmod(OUT, 0o644)
print(f'[saved] {OUT}')

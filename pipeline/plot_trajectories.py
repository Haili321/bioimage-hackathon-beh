"""
Generic trajectory plot from a `trajectories.json` produced by extract_migration*.py.

2 rows x 3 cols, one panel per folder (wt1, wt2, ko1, ko2, ki1, ki2).
Each cell's per-frame centroid trajectory is drawn as a coloured polyline,
with a black dot at the trajectory start and a star at the end. Frame pairs
rejected by the mask-stability filter are drawn in light grey.

Usage:
    python plot_trajectories.py <trajectories_dir> <out_png> [<title_suffix>]

Examples:
    python plot_trajectories.py \\
        /dcs/pg25/u1898019/Desktop/Categorised_Data_finetuned/trajectories \\
        migration_trajectories_finetuned.png \\
        "fine-tuned masks (batch 1)"

    python plot_trajectories.py \\
        /dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_finetuned/trajectories \\
        migration_trajectories_2nd.png \\
        "fine-tuned masks (batch 2, fresh cells)"
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FOLDERS_LAYOUT = [
    ['wt1', 'wt2', 'ko1'],
    ['ko2', 'ki1', 'ki2'],
]
TITLE_MAP = {
    'wt1': 'WT (session 1, low batch)',
    'wt2': 'WT (session 2, high batch)',
    'ko1': 'KO (session 1, low batch)',
    'ko2': 'KO (session 2, high batch)',
    'ki1': 'KI (session 1, low batch)',
    'ki2': 'KI (session 2, high batch)',
}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    traj_dir = Path(sys.argv[1])
    out_png = Path(sys.argv[2])
    title_suffix = sys.argv[3] if len(sys.argv) > 3 else ''

    with open(traj_dir / 'trajectories.json') as f:
        data = json.load(f)

    # Bucket cells by folder
    by_folder = {}
    for cell_id, info in data.items():
        by_folder.setdefault(info['folder'], []).append((cell_id, info))

    # Empty rate from manifest (parent dir)
    manifest_path = traj_dir.parent / 'centering_manifest.json'
    empty_str = ''
    if manifest_path.exists():
        with open(manifest_path) as f:
            m = json.load(f)
        total_T = sum(info['shape'][0] for info in m.values())
        total_e = sum(info['n_empty_masks'] for info in m.values())
        empty_str = f', {total_e}/{total_T} = {100 * total_e / total_T:.2f}% empty'

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for r, row in enumerate(FOLDERS_LAYOUT):
        for c, folder in enumerate(row):
            ax = axes[r, c]
            cells = by_folder.get(folder, [])
            cmap = plt.get_cmap('tab10')
            for i, (cell_id, info) in enumerate(cells):
                traj = np.array(info['traj_um'])  # (T, 2) Δy, Δx
                valid = np.array(info['valid'], dtype=bool)
                pair_valid = np.array(info['pair_valid'], dtype=bool)
                colour = cmap(i % 10)

                # Plot rejected pairs as light grey segments first (so colour overlays on top)
                for t in range(len(pair_valid)):
                    if not pair_valid[t]:
                        ax.plot(traj[t:t+2, 1], traj[t:t+2, 0],
                                color='#cccccc', lw=0.6, zorder=1)
                # Plot valid pairs as coloured segments
                seg_x, seg_y = [], []
                for t in range(len(pair_valid)):
                    if pair_valid[t]:
                        seg_x.extend([traj[t, 1], traj[t+1, 1], None])
                        seg_y.extend([traj[t, 0], traj[t+1, 0], None])
                if seg_x:
                    ax.plot(seg_x, seg_y, color=colour, lw=1.0, alpha=0.85,
                            label=cell_id.split('/')[-1], zorder=2)

                # Start (black dot) and end (coloured star) markers, on first/last valid frame
                if valid.any():
                    valid_idx = np.where(valid)[0]
                    s, e = valid_idx[0], valid_idx[-1]
                    ax.plot(traj[s, 1], traj[s, 0], 'o', color='black', markersize=4, zorder=3)
                    ax.plot(traj[e, 1], traj[e, 0], '*', color=colour,
                            markersize=10, markeredgecolor='black', markeredgewidth=0.5, zorder=3)

            ax.axhline(0, color='#dddddd', lw=0.5)
            ax.axvline(0, color='#dddddd', lw=0.5)
            ax.set_aspect('equal', 'datalim')
            ax.set_title(TITLE_MAP.get(folder, folder), fontsize=10)
            ax.set_xlabel('Δx (μm)', fontsize=9)
            ax.set_ylabel('Δy (μm)', fontsize=9)
            ax.legend(fontsize=7, loc='best', frameon=True)
            ax.grid(alpha=0.2)

    suptitle = f'Cell migration trajectories ({len(data)} cells{empty_str})'
    if title_suffix:
        suptitle += f'.  {title_suffix}'
    fig.suptitle(suptitle + '\nColoured: per-pair mask-stability filter passed.  Light grey: rejected.  Black dot: trajectory start.  Star: end.',
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches='tight')
    print(f'[done] saved {out_png}  ({len(data)} cells across 6 folders)')


if __name__ == '__main__':
    main()

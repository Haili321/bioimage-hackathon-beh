"""
Run Edward's segment_and_center pipeline on all 30 .tif files.
Output: paired *_centered.tif + *_mask.tif + a JSON manifest.

Usage (locally with GPU or via sbatch):
    python process_all_centering.py
"""
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models
from scipy.ndimage import center_of_mass, shift

INPUT_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data')
OUTPUT_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered')
MODEL_TYPE = 'cyto3'          # upgraded from 'cyto' for better low-SNR handling
CHANNELS = [0, 0]              # single-channel grayscale
DIAMETER = None                # let Cellpose auto-estimate
NORM_PERCENTILES = (1, 99)     # per-frame stretching applied only to low-batch files
LOW_BATCH_P99_THRESHOLD = 1000 # files whose 99-th percentile is below this are treated as low-batch
                               # (low batch p99 ~400-900, high batch p99 ~1500-3500)

USE_GPU = torch.cuda.is_available()
print(f'[setup] GPU available: {USE_GPU}', flush=True)
if USE_GPU:
    print(f'[setup] Device: {torch.cuda.get_device_name(0)}', flush=True)

print(f'[setup] Loading Cellpose model ({MODEL_TYPE})...', flush=True)
model = models.Cellpose(model_type=MODEL_TYPE, gpu=USE_GPU)


def normalize_frame(img: np.ndarray, percentiles=NORM_PERCENTILES) -> np.ndarray:
    """Per-frame contrast stretching: clip to [p_lo, p_hi] then rescale to full uint16 range.
    Fixes the low-batch failure where Cellpose cannot see structures with low dynamic range.
    """
    p_lo, p_hi = np.percentile(img, percentiles)
    if p_hi <= p_lo:
        return img.astype(np.uint16)
    norm = np.clip((img.astype(np.float32) - p_lo) / (p_hi - p_lo), 0, 1)
    return (norm * 65535).astype(np.uint16)


def is_low_batch(img4d: np.ndarray) -> bool:
    """Decide whether this stack needs intensity stretching, based on the bulk 99-th percentile.
    Low batch p99 is roughly 400-900, high batch is 1500-3500, so 1000 is a clean separator.
    Subsamples for speed (we only need the rough scale).
    """
    sample = img4d[::5].flatten()[::100]
    return float(np.percentile(sample, 99)) < LOW_BATCH_P99_THRESHOLD


def process_file(input_path: Path, out_centered: Path, out_mask: Path) -> dict:
    """Process one .tif and write paired _centered/_mask files. Returns metadata."""
    img4d = tifffile.imread(input_path)
    T, H, W = img4d.shape
    needs_norm = is_low_batch(img4d)
    print(f'\n[{input_path.name}] T={T}, H={H}, W={W}, dtype={img4d.dtype}, '
          f'normalize={"YES (low batch)" if needs_norm else "NO  (high batch)"}', flush=True)

    centered = np.zeros_like(img4d)
    masks_all = np.zeros((T, H, W), dtype=np.uint8)
    centers = []
    n_empty = 0
    last_center = (H / 2, W / 2)

    t0 = time.time()
    for t in range(T):
        img = img4d[t]
        # Adaptive normalize: only stretch low-batch files (saves high-batch from saturation artefacts)
        img_for_seg = normalize_frame(img) if needs_norm else img
        masks, _, _, _ = model.eval(img_for_seg, diameter=DIAMETER, channels=CHANNELS)

        if masks.max() == 0:
            # Fallback: use last known center, blank mask
            cy, cx = last_center
            mask = np.zeros_like(img, dtype=np.uint8)
            n_empty += 1
        else:
            labels = np.unique(masks)[1:]
            areas = np.array([(masks == l).sum() for l in labels])
            largest = labels[areas.argmax()]
            mask = (masks == largest).astype(np.uint8)
            cy, cx = center_of_mass(mask)
            last_center = (cy, cx)

        centers.append([float(cy), float(cx)])

        shift_y, shift_x = H / 2 - cy, W / 2 - cx
        centered[t] = shift(img, shift=(shift_y, shift_x), mode='nearest')
        masks_all[t] = mask

        if (t + 1) % 50 == 0 or t + 1 == T:
            print(f'  [{input_path.name}] frame {t+1}/{T}  ({(time.time()-t0)/(t+1):.2f}s/frame)', flush=True)

    elapsed = time.time() - t0
    print(f'  [{input_path.name}] DONE in {elapsed:.1f}s ({elapsed/T:.2f}s/frame), '
          f'empty masks: {n_empty}/{T}', flush=True)

    out_centered.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_centered, centered)
    tifffile.imwrite(out_mask, masks_all * 255)  # scale 0/1 to 0/255 for Fiji visibility

    return {
        'input': str(input_path),
        'centered': str(out_centered),
        'mask': str(out_mask),
        'shape': [int(T), int(H), int(W)],
        'dtype': str(img4d.dtype),
        'centers': centers,
        'n_empty_masks': int(n_empty),
        'elapsed_s': float(elapsed),
        'sec_per_frame': float(elapsed / T),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.rglob('*.tif'))
    print(f'[main] Found {len(input_files)} input files', flush=True)

    manifest = {}
    t_total = time.time()
    for i, f in enumerate(input_files, 1):
        rel = f.relative_to(INPUT_DIR)
        out_centered = OUTPUT_DIR / rel.parent / f'{f.stem}_centered.tif'
        out_mask = OUTPUT_DIR / rel.parent / f'{f.stem}_mask.tif'

        if out_centered.exists() and out_mask.exists():
            print(f'[skip] {rel} already processed', flush=True)
            continue

        print(f'\n[{i}/{len(input_files)}] {rel}', flush=True)
        info = process_file(f, out_centered, out_mask)
        manifest[str(rel)] = info

        # Save manifest incrementally so partial runs are recoverable
        with open(OUTPUT_DIR / 'centering_manifest.json', 'w') as fout:
            json.dump(manifest, fout, indent=2)

    print(f'\n[DONE] Total: {time.time()-t_total:.0f}s', flush=True)
    print(f'[DONE] Output: {OUTPUT_DIR}', flush=True)


if __name__ == '__main__':
    main()

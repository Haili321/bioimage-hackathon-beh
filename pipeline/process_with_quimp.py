"""
Run the centering pipeline using the Quimp-trained Cellpose model.

Same as process_with_finetuned.py but loads the model from
training_data_quimp/models/ and writes outputs to a parallel directory.
The existing self-trained outputs and model are not touched.

Processes BOTH batches:
    Categorised_Data       -> Categorised_Data_quimp_finetuned
    2ndUpload              -> Categorised_Data_2nd_quimp_finetuned
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose import models
from scipy.ndimage import center_of_mass, shift

TRAIN_DIR = Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_centered/training_data_quimp')

JOBS = [
    (Path('/dcs/pg25/u1898019/Desktop/Categorised_Data'),
     Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_quimp_finetuned')),
    (Path('/dcs/pg25/u1898019/Desktop/2ndUpload'),
     Path('/dcs/pg25/u1898019/Desktop/Categorised_Data_2nd_quimp_finetuned')),
]


def find_finetuned_model() -> Path:
    candidates = sorted((TRAIN_DIR / 'models').glob('*'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'No model found in {TRAIN_DIR / "models"}')
    return candidates[-1]


CHANNELS = [0, 0]
DIAMETER = None
NORM_PERCENTILES = (1, 99)
LOW_BATCH_P99_THRESHOLD = 1000


def normalize_frame(img):
    p_lo, p_hi = np.percentile(img, NORM_PERCENTILES)
    if p_hi <= p_lo:
        return img.astype(np.uint16)
    norm = np.clip((img.astype(np.float32) - p_lo) / (p_hi - p_lo), 0, 1)
    return (norm * 65535).astype(np.uint16)


def is_low_batch(img4d):
    sample = img4d[::5].flatten()[::100]
    return float(np.percentile(sample, 99)) < LOW_BATCH_P99_THRESHOLD


USE_GPU = torch.cuda.is_available()
print(f'[setup] GPU available: {USE_GPU}', flush=True)
if USE_GPU:
    print(f'[setup] Device: {torch.cuda.get_device_name(0)}', flush=True)

model_path = find_finetuned_model()
print(f'[setup] Loading Quimp-trained model: {model_path}', flush=True)
model = models.CellposeModel(pretrained_model=str(model_path), gpu=USE_GPU)


def process_file(input_path, out_centered, out_mask):
    img4d = tifffile.imread(input_path)
    T, H, W = img4d.shape
    needs_norm = is_low_batch(img4d)
    print(f'\n[{input_path.name}] T={T}, H={H}, W={W}, normalize={"YES" if needs_norm else "NO "}', flush=True)

    centered = np.zeros_like(img4d)
    masks_all = np.zeros((T, H, W), dtype=np.uint8)
    centers = []
    n_empty = 0
    last_center = (H / 2, W / 2)

    t0 = time.time()
    for t in range(T):
        img = img4d[t]
        img_for_seg = normalize_frame(img) if needs_norm else img
        result = model.eval(img_for_seg, diameter=DIAMETER, channels=CHANNELS)
        masks = result[0]

        if masks.max() == 0:
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
            print(f'  frame {t+1}/{T} ({(time.time()-t0)/(t+1):.2f}s/frame)', flush=True)

    elapsed = time.time() - t0
    print(f'  DONE in {elapsed:.1f}s, empty masks: {n_empty}/{T}', flush=True)

    out_centered.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_centered, centered)
    tifffile.imwrite(out_mask, masks_all * 255)

    return {
        'input': str(input_path),
        'centered': str(out_centered),
        'mask': str(out_mask),
        'shape': [int(T), int(H), int(W)],
        'centers': centers,
        'n_empty_masks': int(n_empty),
        'elapsed_s': float(elapsed),
        'sec_per_frame': float(elapsed / T),
        'model_used': str(model_path.name),
    }


def main():
    for input_dir, output_dir in JOBS:
        print(f'\n=== JOB: {input_dir.name} -> {output_dir.name} ===', flush=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_files = sorted(input_dir.rglob('*.tif'))
        print(f'[main] {len(input_files)} input files', flush=True)

        manifest = {}
        t_total = time.time()
        for i, f in enumerate(input_files, 1):
            rel = f.relative_to(input_dir)
            out_centered = output_dir / rel.parent / f'{f.stem}_centered.tif'
            out_mask = output_dir / rel.parent / f'{f.stem}_mask.tif'

            if out_centered.exists() and out_mask.exists():
                print(f'[skip] {rel} already processed', flush=True)
                continue

            print(f'\n[{i}/{len(input_files)}] {rel}', flush=True)
            info = process_file(f, out_centered, out_mask)
            manifest[str(rel)] = info

            with open(output_dir / 'centering_manifest.json', 'w') as fout:
                json.dump(manifest, fout, indent=2)

        print(f'\n[done] {input_dir.name}: {time.time() - t_total:.0f}s', flush=True)


if __name__ == '__main__':
    main()

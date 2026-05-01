import os
import numpy as np
import tifffile as tiff
import pandas as pd
from scipy.ndimage import center_of_mass

# ---- your modules ----
from segment_and_center import segment_and_center_timeseries
from otsu_threshold import (
    refine_body_and_lamellipodia,
    align_and_apply_masks,
    separate_body_lamellipodia_percentile
)
from cosine_analysis import (
    compute_movement_vectors_windowed,
    compute_polarity_vectors,
    cosine_similarity_rows
)

# optional (if using leading edge)
from lamellipodia_analysis import (
    split_front_back_lamellipodia_movement
)


# -------------------------
# Helper: velocity
# -------------------------
def compute_velocity(movement_vectors):
    return [
        np.sqrt(dy**2 + dx**2) if not np.isnan(dy) else np.nan
        for dy, dx in movement_vectors
    ]


# -------------------------
# Process single file
# -------------------------
def process_file(input_path, output_dir,
                 dilation=5,
                 percentile=70,
                 window=3):

    name = os.path.splitext(os.path.basename(input_path))[0]
    save_dir = os.path.join(output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Processing: {name}")

    # ---- Load ----
    images = tiff.imread(input_path)

    # ---- Pipeline ----
    centered_images, masks, centers = segment_and_center_timeseries(images, model_type="cellpose_1777590610.8133328")

    aligned_masks, cleaned_images = align_and_apply_masks(
        centered_images, masks, centers, dilation_radius=dilation
    )

    body_masks, lam_masks, _ = separate_body_lamellipodia_percentile(
        cleaned_images, aligned_masks, body_percentile=percentile
    )

    body_masks, lam_masks = refine_body_and_lamellipodia(
        body_masks, lam_masks
    )

    # ---- Centers ----
    body_centers = [center_of_mass(b) for b in body_masks]
    lam_centers = [center_of_mass(l) for l in lam_masks]

    # ---- Vectors ----
    polarity_vectors = compute_polarity_vectors(body_centers, lam_centers)
    movement_vectors = compute_movement_vectors_windowed(body_centers, window)

    cosine_sim = cosine_similarity_rows(np.array(polarity_vectors), np.array(movement_vectors))
    velocity = compute_velocity(movement_vectors)

    # ---- Leading edge (movement-based) ----
    front_mov, back_mov = split_front_back_lamellipodia_movement(
        lam_masks, body_centers, movement_vectors
    )

    front_area = [np.sum(m) for m in front_mov]
    back_area = [np.sum(m) for m in back_mov]

    # -------------------------
    # Save masks
    # -------------------------
    tiff.imwrite(os.path.join(save_dir, "body_masks.tif"), body_masks.astype(np.uint8))
    tiff.imwrite(os.path.join(save_dir, "lam_masks.tif"), lam_masks.astype(np.uint8))
    tiff.imwrite(os.path.join(save_dir, "front_masks.tif"), front_mov.astype(np.uint8))

    # -------------------------
    # Save cleaned images (optional)
    # -------------------------
    tiff.imwrite(os.path.join(save_dir, "cleaned_images.tif"), cleaned_images.astype(np.float32))

    # -------------------------
    # Save metrics
    # -------------------------
    df = pd.DataFrame({
        "frame": np.arange(len(body_masks)),
        "body_cy": [c[0] for c in body_centers],
        "body_cx": [c[1] for c in body_centers],
        "polarity_dy": [v[0] for v in polarity_vectors],
        "polarity_dx": [v[1] for v in polarity_vectors],
        "movement_dy": [v[0] for v in movement_vectors],
        "movement_dx": [v[1] for v in movement_vectors],
        "velocity": velocity,
        "cosine_similarity": cosine_sim,
        "front_area": front_area,
        "back_area": back_area
    })

    df.to_csv(os.path.join(save_dir, "metrics.csv"), index=False)

    print(f"Saved: {save_dir}")

def process_folder(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    files = [
        f for f in os.listdir(input_folder)
        if f.endswith(".tif") or f.endswith(".tiff")
    ]

    all_data = []

    for f in files:
        input_path = os.path.join(input_folder, f)

        try:
            df = process_file(input_path, output_folder)
            all_data.append(df)

        except Exception as e:
            print(f"Failed: {f} | Error: {e}")
            continue

    # -------------------------
    # GLOBAL SUMMARY
    # -------------------------
    if len(all_data) > 0:

        global_df = pd.concat(all_data, ignore_index=True)

        summary_path = os.path.join(output_folder, "global_summary.csv")
        global_df.to_csv(summary_path, index=False)

        print(f"\nSaved global summary: {summary_path}")

process_folder(
    input_folder="data/",
    output_folder="results/"
)
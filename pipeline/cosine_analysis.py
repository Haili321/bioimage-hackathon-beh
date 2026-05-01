import csv, math
import tifffile as tiff
import numpy as np
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt



def compute_centers_of_mass(cell_body_masks, lamellipodia_masks):
    """
    Compute center of mass for body and lamellipodia masks.

    Parameters:
        cell_body_masks (np.ndarray): (T, H, W)
        lamellipodia_masks (np.ndarray): (T, H, W)

    Returns:
        body_centers (list of tuples)
        lam_centers (list of tuples)
    """

    body_centers = []
    lam_centers = []

    for body, lam in zip(cell_body_masks, lamellipodia_masks):

        # Body center
        if np.any(body):
            body_com = center_of_mass(body)
        else:
            body_com = (np.nan, np.nan)

        # Lamellipodia center
        if np.any(lam):
            lam_com = center_of_mass(lam)
        else:
            lam_com = (np.nan, np.nan)

        body_centers.append(body_com)
        lam_centers.append(lam_com)

    return body_centers, lam_centers
def read_tiff_stack(path):
    """
    Read a TIFF file (single image or time series).

    Parameters:
        path (str): path to .tif/.tiff file

    Returns:
        np.ndarray: image array (T, H, W) or (H, W)
    """
    img = tiff.imread(path)
    return img

def compute_polarity_vectors(body_centers, lam_centers):
    """
    Compute vectors from body → lamellipodia.

    Returns:
        vectors: list of (dy, dx)
    """
    vectors = []

    for b, l in zip(body_centers, lam_centers):
        if np.any(np.isnan(b)) or np.any(np.isnan(l)):
            vectors.append((np.nan, np.nan))
        else:
            dy = l[0] - b[0]
            dx = l[1] - b[1]
            vectors.append((dy, dx))

    return vectors



def plot_vectors_with_cosine(
    images,
    body_masks,
    lam_masks,
    body_centers,
    polarity_vectors,
    movement_vectors,
    cosine_similarities,
    n_cols=5,
    scale_pol=1.5,
    scale_mov=3.0
):
    n = len(images)
    n_rows = math.ceil(n / n_cols)

    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for i in range(n):
        img = images[i]
        body = body_masks[i]
        lam = lam_masks[i]

        cy, cx = body_centers[i]
        dy_pol, dx_pol = polarity_vectors[i]
        dy_mov, dx_mov = movement_vectors[i]
        cos_sim = cosine_similarities[i]

        # normalize image
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # overlay
        overlay = np.zeros((*img.shape, 3))
        #overlay[..., 0] = body.astype(float)
        #overlay[..., 2] = lam.astype(float)
        overlay[..., 1] = img_norm * 0.5

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(overlay)

        # polarity vector (yellow)
        if not np.isnan(dy_pol):
            plt.arrow(cx, cy, dx_pol * scale_pol, dy_pol * scale_pol,
                      color='yellow', head_width=5)

        # movement vector (green)
        if not np.isnan(dy_mov):
            plt.arrow(cx, cy, dx_mov * scale_mov, dy_mov * scale_mov,
                      color='lime', head_width=5)

        # --- cosine similarity label ---
        if not np.isnan(cos_sim):
            label = f"{cos_sim:.2f}"
        else:
            label = "nan"

        plt.title(f"Frame {i} | cos={label}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def compute_movement_vectors(body_centers):
    """
    Compute frame-to-frame movement vectors from body centers.

    Returns:
        movement_vectors: list of (dy, dx)
    """
    movement_vectors = []

    for i in range(len(body_centers)):
        if i == 0:
            movement_vectors.append((0, 0))  # no movement for first frame
            continue

        prev = body_centers[i - 1]
        curr = body_centers[i]

        if np.any(np.isnan(prev)) or np.any(np.isnan(curr)):
            movement_vectors.append((np.nan, np.nan))
        else:
            dy = curr[0] - prev[0]
            dx = curr[1] - prev[1]
            movement_vectors.append((dy, dx))

    return movement_vectors

def compute_movement_vectors_windowed(body_centers, window=3):
    """
    Compute movement vectors using a temporal window.

    Parameters:
        body_centers: list of (y, x)
        window (int): number of frames to average over

    Returns:
        movement_vectors: list of (dy, dx)
    """

    movement_vectors = []
    n = len(body_centers)

    for i in range(n):

        # define window range
        j = min(i + window, n - 1)

        c0 = body_centers[i]
        c1 = body_centers[j]

        if np.any(np.isnan(c0)) or np.any(np.isnan(c1)):
            movement_vectors.append((np.nan, np.nan))
        else:
            dy = (c1[0] - c0[0]) / (j - i if j != i else 1)
            dx = (c1[1] - c0[1]) / (j - i if j != i else 1)
            movement_vectors.append((dy, dx))

    return movement_vectors

def cosine_similarity_rows(A, B):
    # dot product for each row
    dot = np.sum(A * B, axis=1)

    # norms for each row
    norm_A = np.linalg.norm(A, axis=1)
    norm_B = np.linalg.norm(B, axis=1)

    return dot / (norm_A * norm_B)

from scipy.ndimage import shift

def uncenter_images(centered_images, centers):
    """
    Map centered images back to original coordinates.

    Parameters:
        centered_images (np.ndarray): (T, H, W)
        centers (list): original (cy, cx)

    Returns:
        uncentered_images (np.ndarray)
    """

    H, W = centered_images[0].shape
    uncentered = []

    for img, (cy, cx) in zip(centered_images, centers):

        shift_y = H / 2 - cy
        shift_x = W / 2 - cx

        # invert the shift
        img_orig = shift(
            img,
            shift=(-shift_y, -shift_x),
            mode='constant',
            cval=0
        )

        uncentered.append(img_orig)

    return np.array(uncentered)


def detect_turning_events_from_movement(movement_vectors, angle_threshold=np.deg2rad(30)):
    """
    Turning = large change in movement direction between consecutive frames.

    Returns:
        turning_events: binary array
        angles: movement angles
        angle_changes: angular differences
    """

    angles = []
    turning_events = [0]
    angle_changes = [0]

    prev_angle = None

    for dy, dx in movement_vectors:

        if np.isnan(dy) or np.isnan(dx):
            angles.append(np.nan)
            turning_events.append(0)
            angle_changes.append(np.nan)
            continue

        angle = np.arctan2(dy, dx)
        angles.append(angle)

        if prev_angle is None:
            turning_events.append(0)
            angle_changes.append(0)
        else:
            # wrapped angular difference
            dtheta = np.arctan2(
                np.sin(angle - prev_angle),
                np.cos(angle - prev_angle)
            )

            angle_changes.append(dtheta)

            turning_events.append(
                1 if np.abs(dtheta) > angle_threshold else 0
            )

        prev_angle = angle

    return turning_events[1:], angles, angle_changes[1:]


'''path = "ki_25"
lamellipodia = read_tiff_stack("data/output/%s/lamellipodia.tif"%(path))
cell_bodies = read_tiff_stack("data/output/%s/cell_body.tif"%(path))
cleaned_images = read_tiff_stack("data/output/%s/cleaned_images.tif"%(path))
with open("data/output/%s/centers.csv"%(path), "r") as file:
    r = csv.reader(file)
    centers = list(r)

body_centers, lam_centers = compute_centers_of_mass(
    cell_bodies,
    lamellipodia
)
print(body_centers)
print(centers)

centers = centers[0]
centers = [eval(x) for x in centers]
vectors = compute_polarity_vectors(body_centers, lam_centers)


movement_vectors = compute_movement_vectors_windowed(centers)

lower=30
upper=40

similarity = cosine_similarity_rows(np.array(vectors), np.array(movement_vectors))
print(similarity)

plot_vectors_with_cosine(
    cleaned_images[lower:upper],
    cell_bodies[lower:upper],
    lamellipodia[lower:upper],
    body_centers[lower:upper],
    vectors[lower:upper],
    movement_vectors[lower:upper],
    cosine_similarities=similarity[lower:upper],
    n_cols=5
)



plt.figure()
plt.bar(np.arange(len(similarity)), similarity)
plt.xlabel("Vector index")
plt.ylabel("Cosine similarity")
plt.title("Row-wise Cosine Similarity")
plt.show()'''
import tifffile as tiff
import csv, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, shift, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk


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


def align_and_apply_masks(centered_images, masks, centers, dilation_radius=1):
    """
    Align masks to centered images using stored centroids, then apply them.

    Parameters:
        centered_images (np.ndarray): (T, H, W)
        masks (list or np.ndarray): original masks (T, H, W)
        centers (list): [(cy, cx), ...] from original images
        dilation_radius (int): expand mask to include edge uncertainty

    Returns:
        aligned_masks (np.ndarray)
        cleaned_images (np.ndarray)
    """

    aligned_masks = []
    cleaned_images = []

    H, W = centered_images[0].shape

    for img, mask, (cy, cx) in zip(centered_images, masks, centers):

        # Compute the SAME shift used during centering
        shift_y = H / 2 - cy
        shift_x = W / 2 - cx

        # Shift mask (order=0 = nearest neighbor, preserves labels)
        shifted_mask = shift(
            mask,
            shift=(shift_y, shift_x),
            order=0,
            mode='constant',
            cval=0
        )

        # Binarize (safety)
        shifted_mask = (shifted_mask > 0)

        # Dilate mask slightly
        dilated_mask = binary_dilation(shifted_mask, iterations=dilation_radius)

        # Apply mask to centered image
        cleaned = img * dilated_mask

        aligned_masks.append(dilated_mask)
        cleaned_images.append(cleaned)

    return np.array(aligned_masks), np.array(cleaned_images)

def show_image_grid(images, cols=5):
    """
    Display a stack of images in a grid.

    Parameters:
        images (np.ndarray): (T, H, W)
        cols (int): number of columns
    """
    n = len(images)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 3, rows * 3))

    for i, img in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Frame {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def separate_cell_body_lamellipodia(cleaned_images, aligned_masks):
    """
    Apply Otsu threshold to separate main cell body from lamellipodia.

    Parameters:
        cleaned_images (np.ndarray): (T, H, W)
        aligned_masks (np.ndarray): (T, H, W) boolean

    Returns:
        cell_body_masks (np.ndarray)
        lamellipodia_masks (np.ndarray)
        thresholds (list)
    """

    cell_body_masks = []
    lamellipodia_masks = []
    thresholds = []

    for img, mask in zip(cleaned_images, aligned_masks):

        # Extract only cell pixels
        pixels = img[mask > 0]

        if len(pixels) == 0:
            # fallback if something went wrong
            cell_body = np.zeros_like(mask)
            lamellipodia = np.zeros_like(mask)
            thresh = 0

        else:
            # Compute Otsu threshold on cell region only
            thresh = threshold_otsu(pixels)

            # Main body = brighter region
            cell_body = (img >= thresh) & mask

            # Lamellipodia = dimmer region within mask
            lamellipodia = (img < thresh) & mask

        cell_body_masks.append(cell_body)
        lamellipodia_masks.append(lamellipodia)
        thresholds.append(thresh)

    return (
        np.array(cell_body_masks),
        np.array(lamellipodia_masks),
        thresholds
    )


def show_body_lamellipodia_overlays(
    images,
    cell_body_masks,
    lamellipodia_masks,
    n_cols=5
):
    """
    Overlay cell body and lamellipodia on images.

    Parameters:
        images: (T, H, W)
        cell_body_masks: (T, H, W)
        lamellipodia_masks: (T, H, W)
    """

    n = len(images)
    n_rows = int(np.ceil(n / n_cols))

    plt.figure(figsize=(n_cols * 3, n_rows * 3))

    for i in range(n):
        img = images[i]
        body = cell_body_masks[i]
        lam = lamellipodia_masks[i]

        # Normalize image for display
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # RGB overlay
        overlay = np.zeros((*img.shape, 3))
        overlay[..., 0] = body      # red channel = cell body
        overlay[..., 2] = lam       # blue channel = lamellipodia
        overlay[..., 1] = img_norm * 0.6  # green = background intensity

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(overlay)
        plt.title(f"Frame {i}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def separate_body_lamellipodia_percentile(
    images,
    masks,
    body_percentile=50
):
    """
    Separate cell body vs lamellipodia using intensity percentile threshold.

    Parameters:
        images (np.ndarray): (T, H, W)
        masks (np.ndarray): (T, H, W)
        body_percentile (float): % of brightest pixels considered "cell body"

    Returns:
        cell_body_masks, lamellipodia_masks, thresholds
    """

    cell_body_masks = []
    lamellipodia_masks = []
    thresholds = []

    for img, mask in zip(images, masks):

        pixels = img[mask > 0]

        if len(pixels) == 0:
            cell_body = np.zeros_like(mask)
            lam = np.zeros_like(mask)
            thresh = 0

        else:
            # percentile threshold instead of Otsu
            thresh = np.percentile(pixels, body_percentile)

            cell_body = (img >= thresh) & mask
            lam = (img < thresh) & mask

        cell_body_masks.append(cell_body)
        lamellipodia_masks.append(lam)
        thresholds.append(thresh)

    return (
        np.array(cell_body_masks),
        np.array(lamellipodia_masks),
        thresholds
    )



def refine_body_and_lamellipodia(
    cell_body_masks,
    lamellipodia_masks,
    closing_radius=3
):
    """
    Refine existing body + lamellipodia masks:
    - clean holes in body
    - enforce lamellipodia = original_mask - cleaned_body

    Parameters:
        cell_body_masks (np.ndarray): (T, H, W)
        lamellipodia_masks (np.ndarray): (T, H, W)
        closing_radius (int): morphological smoothing strength

    Returns:
        cleaned_body
        cleaned_lamellipodia
    """

    cleaned_body = []
    cleaned_lam = []

    selem = disk(closing_radius)

    for body, lam in zip(cell_body_masks, lamellipodia_masks):

        body = body.astype(bool)
        lam = lam.astype(bool)

        # --- 1. Clean body mask ---
        body_closed = binary_closing(body, selem)
        body_filled = binary_fill_holes(body_closed)

        # --- 2. Enforce consistency ---
        # lamellipodia should NOT overlap cleaned body
        lam_clean = lam & ~body_filled

        cleaned_body.append(body_filled)
        cleaned_lam.append(lam_clean)

    return (
        np.array(cleaned_body),
        np.array(cleaned_lam)
    )

'''path = "ki_25"

centered_images = read_tiff_stack("data/output/%s/centered_stack.tif"%(path))
masks_all = read_tiff_stack("data/output/%s/mask_stack.tif"%(path))
with open("data/output/%s/centers.csv"%(path), "r") as file:
    r = csv.reader(file)
    centers = list(r)




centers = centers[0]
centers = [eval(x) for x in centers]
aligned_masks, cleaned_images = align_and_apply_masks(centered_images, masks_all, centers)

cell_body, lamellipodia, thresholds = separate_body_lamellipodia_percentile(cleaned_images, aligned_masks, body_percentile=60)
#cell_body, lamellipodia, thresholds = separate_cell_body_lamellipodia(cleaned_images, aligned_masks)


cleaned_body, cleaned_lam = refine_body_and_lamellipodia(cell_body, lamellipodia, closing_radius=50)


lower = 20
upper = 45
show_body_lamellipodia_overlays(
    cleaned_images[lower:upper],
    cleaned_body[lower:upper],
    cleaned_lam[lower:upper],
    n_cols=5
)


tiff.imwrite("data/output/%s/cell_body.tif"%(path), cleaned_body.astype('uint16'))
tiff.imwrite("data/output/%s/lamellipodia.tif"%(path), cleaned_lam.astype('uint16'))
tiff.imwrite("data/output/%s/cleaned_images.tif"%(path), cleaned_images.astype('uint16'))'''

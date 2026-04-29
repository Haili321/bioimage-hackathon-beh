import tifffile as tiff
import csv, math
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, shift
from skimage.filters import threshold_otsu

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


def align_and_apply_masks(centered_images, masks, centers, dilation_radius=5):
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


centered_images = read_tiff_stack("data/output/wt_5/centered_stack.tif")
masks_all = read_tiff_stack("data/output/wt_5/mask_stack.tif")
with open("data/output/wt_5/centers.csv", "r") as file:
    r = csv.reader(file)
    centers = list(r)


centers = centers[0]
centers = [eval(x) for x in centers]
aligned_masks, cleaned_images = align_and_apply_masks(centered_images, masks_all, centers)



cell_body, lamellipodia, thresholds = separate_cell_body_lamellipodia(cleaned_images, aligned_masks)

show_image_grid(lamellipodia[20:45])
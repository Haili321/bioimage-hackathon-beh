import numpy as np
from cellpose import models
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt
import math
import tifffile as tiff
import csv

def segment_and_center_timeseries(images, model_type='cpsam', diameter=None, channels=[0, 0]):
    """
    Segment and center a single-cell time series using Cellpose.

    Parameters:
        images (list or np.ndarray): list/array of 2D images (T, H, W)
        model_type (str): 'cyto', 'nuclei', or custom Cellpose model
        diameter (float or None): approximate cell diameter
        channels (list): Cellpose channel setting

    Returns:
        centered_images (np.ndarray): centered image stack
        masks (list): segmentation masks
        centers (list): original centroids
    """

    model = models.CellposeModel(pretrained_model=model_type, gpu=True)

    centered_images = []
    masks_all = []
    centers = []

    for img in images:
        masks, flows, styles = model.eval(
            img,
            diameter=diameter,
            flow_threshold=0.4,
            normalize=True
        )

        # If multiple masks, keep the largest (safe fallback)
        if masks.max() > 1:
            labels = np.unique(masks)[1:]
            areas = [(masks == l).sum() for l in labels]
            largest_label = labels[np.argmax(areas)]
            mask = (masks == largest_label).astype(np.uint8)
        else:
            mask = (masks > 0).astype(np.uint8)

        # Compute centroid
        cy, cx = center_of_mass(mask)
        centers.append((cy, cx))

        # Compute shift to center
        H, W = img.shape
        shift_y = H / 2 - cy
        shift_x = W / 2 - cx

        # Apply shift
        centered = shift(img, shift=(shift_y, shift_x), mode='nearest')

        centered_images.append(centered)
        masks_all.append(mask)

    return np.array(centered_images), np.array(masks_all), centers



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


'''path = "ki_14"

images = read_tiff_stack("data/%s.tif"%(path))

centered_images, masks_all, centers = segment_and_center_timeseries(images, model_type="cpsam")

temp = []
for x in centers:
    temp.append((float(x[0]), float(x[1])))

centers = temp

tiff.imwrite("data/output/%s/centered_stack.tif"%(path), centered_images.astype('float32'))
tiff.imwrite("data/output/%s/mask_stack.tif"%(path), masks_all.astype('uint16'))
with open("data/output/%s/centers.csv"%(path), "w") as file:
    wr = csv.writer(file)
    wr.writerow(centers)'''
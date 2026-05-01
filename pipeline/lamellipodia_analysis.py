import numpy as np

def split_front_back_lamellipodia(
    lam_masks,
    body_centers,
    polarity_vectors
):
    """
    Split lamellipodia into front/back using polarity vector.

    Parameters:
        lam_masks (np.ndarray): (T, H, W)
        body_centers (list): [(cy, cx), ...]
        polarity_vectors (list): [(dy, dx), ...]

    Returns:
        front_masks, back_masks
    """

    front_masks = []
    back_masks = []

    for lam, (cy, cx), (dy, dx) in zip(
        lam_masks, body_centers, polarity_vectors
    ):

        lam = lam.astype(bool)

        if np.isnan(dy) or np.isnan(dx):
            front_masks.append(np.zeros_like(lam))
            back_masks.append(lam)
            continue

        # normalize polarity vector
        norm = np.sqrt(dy**2 + dx**2) + 1e-8
        vy, vx = dy / norm, dx / norm

        # grid of coordinates
        yy, xx = np.indices(lam.shape)

        # vectors from center to each pixel
        rel_y = yy - cy
        rel_x = xx - cx

        # dot product
        dot = rel_y * vy + rel_x * vx

        # split
        front = (dot > 0) & lam
        back = (dot <= 0) & lam

        front_masks.append(front)
        back_masks.append(back)

    return np.array(front_masks), np.array(back_masks)


def split_front_back_lamellipodia_movement(
    lam_masks,
    body_centers,
    movement_vectors,
    min_magnitude=1e-3
):
    """
    Split lamellipodia into front/back using movement direction.

    Parameters:
        lam_masks (np.ndarray): (T, H, W)
        body_centers (list): [(cy, cx), ...]
        movement_vectors (list): [(dy, dx), ...]
        min_magnitude (float): threshold to ignore tiny movement

    Returns:
        front_masks, back_masks
    """

    front_masks = []
    back_masks = []

    for lam, (cy, cx), (dy, dx) in zip(
        lam_masks, body_centers, movement_vectors
    ):

        lam = lam.astype(bool)

        # handle invalid or tiny movement
        mag = np.sqrt(dy**2 + dx**2)
        if np.isnan(dy) or np.isnan(dx) or mag < min_magnitude:
            front_masks.append(np.zeros_like(lam))
            back_masks.append(lam)
            continue

        # normalize movement vector
        vy, vx = dy / mag, dx / mag

        yy, xx = np.indices(lam.shape)

        rel_y = yy - cy
        rel_x = xx - cx

        dot = rel_y * vy + rel_x * vx

        front = (dot > 0) & lam
        back = (dot <= 0) & lam

        front_masks.append(front)
        back_masks.append(back)

    return np.array(front_masks), np.array(back_masks)

import numpy as np

def compute_leading_edge_tip(
    lam_masks,
    body_centers,
    movement_vectors
):
    """
    Find the lamellipodia pixel furthest in movement direction.

    Returns:
        tip_positions: [(y, x), ...]
    """

    tips = []

    for lam, (cy, cx), (dy, dx) in zip(
        lam_masks, body_centers, movement_vectors
    ):
        lam = lam.astype(bool)

        mag = np.sqrt(dy**2 + dx**2)

        if np.isnan(dy) or mag < 1e-6 or not np.any(lam):
            tips.append((np.nan, np.nan))
            continue

        # normalize direction
        vy, vx = dy / mag, dx / mag

        yy, xx = np.where(lam)

        rel_y = yy - cy
        rel_x = xx - cx

        dot = rel_y * vy + rel_x * vx

        idx = np.argmax(dot)

        tip_y = yy[idx]
        tip_x = xx[idx]

        tips.append((tip_y, tip_x))

    return tips



def compute_tip_velocity(tips):
    velocities = []

    for i in range(len(tips)):
        if i == 0:
            velocities.append(np.nan)
            continue

        y0, x0 = tips[i - 1]
        y1, x1 = tips[i]

        if np.isnan(y0) or np.isnan(y1):
            velocities.append(np.nan)
        else:
            dy = y1 - y0
            dx = x1 - x0
            velocities.append(np.sqrt(dy**2 + dx**2))

    return velocities
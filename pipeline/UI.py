import streamlit as st
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from segment_and_center import segment_and_center_timeseries
from otsu_threshold import refine_body_and_lamellipodia, align_and_apply_masks, separate_body_lamellipodia_percentile
from cosine_analysis import compute_movement_vectors_windowed, compute_polarity_vectors, cosine_similarity_rows, uncenter_images, detect_turning_events_from_movement
from scipy.ndimage import center_of_mass
from lamellipodia_analysis import split_front_back_lamellipodia, split_front_back_lamellipodia_movement, compute_leading_edge_tip, compute_tip_velocity

st.title("Cell Migration Analysis")

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload TIFF", type=["tif", "tiff"])

# -------------------------
# Parameters
# -------------------------
st.sidebar.header("Parameters")

window = st.sidebar.slider("Movement window", 1, 10, 3)
dilation = st.sidebar.slider("Mask dilation", 1, 10, 1)
percentile = st.sidebar.slider("Body percentile", 50, 95, 70)
closure = st.sidebar.slider("Morphological closure", 1, 50, 10)

# -------------------------
# Run pipeline
# -------------------------
@st.cache_data
def run_pipeline(images, dilation, percentile, window, closure):

    centered_images, masks, centers = segment_and_center_timeseries(images, model_type='cellpose_1777590610.8133328')

    aligned_masks, cleaned_images = align_and_apply_masks(
        centered_images, masks, centers, dilation_radius=dilation
    )

    body_masks, lam_masks, _ = separate_body_lamellipodia_percentile(
        cleaned_images, aligned_masks, body_percentile=percentile
    )

    body_masks, lam_masks = refine_body_and_lamellipodia(
        body_masks, lam_masks, closing_radius=closure
    )

    body_centers = [center_of_mass(b) for b in body_masks]
    lam_centers = [center_of_mass(l) for l in lam_masks]

    polarity_vectors = compute_polarity_vectors(body_centers, lam_centers)
    movement_vectors = compute_movement_vectors_windowed(centers, window)

    cosine_sim = cosine_similarity_rows(np.array(polarity_vectors), np.array(movement_vectors))

    return {
        "images": cleaned_images,
        "body_masks": body_masks,
        "lam_masks": lam_masks,
        "body_centers": body_centers,
        "polarity_vectors": polarity_vectors,
        "movement_vectors": movement_vectors,
        "cosine_sim": cosine_sim,
        "original_centers": centers
    }

if uploaded_file is not None:

    images = tiff.imread(uploaded_file)
    st.write(f"Loaded shape: {images.shape}")

    if st.button("Run Analysis"):
        with st.spinner("Running pipeline..."):
            st.session_state.results = run_pipeline(
                images, dilation, percentile, window, closure
            )
        st.success("Analysis complete!")

# -------------------------
# Visualization (persistent)
# -------------------------
if "results" in st.session_state:

    data = st.session_state.results

    images = data["images"]
    body_masks = data["body_masks"]
    lam_masks = data["lam_masks"]
    body_centers = data["body_centers"]
    polarity_vectors = data["polarity_vectors"]
    movement_vectors = data["movement_vectors"]
    cosine_sim = data["cosine_sim"]

    st.subheader("Visualization")

    frame = st.slider("Frame", 0, len(images) - 1, 0)
    st.session_state.current_frame = frame

    img = images[frame]
    body = body_masks[frame]
    lam = lam_masks[frame]

    cy, cx = body_centers[frame]
    dy_pol, dx_pol = polarity_vectors[frame]
    dy_mov, dx_mov = movement_vectors[frame]
    cos_val = cosine_sim[frame]

    # normalize image
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # overlay image
    overlay = np.zeros((*img.shape, 3))
    overlay[..., 0] = body.astype(float)        # red = body
    overlay[..., 2] = lam.astype(float)         # blue = lamellipodia
    overlay[..., 1] = img_norm * 0.5            # green = background

    # ---- SIDE-BY-SIDE PLOT ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    original_centers = data["original_centers"]

    uncentered_images = uncenter_images(images, original_centers)


    # LEFT: cleaned grayscale
    axes[0].imshow(uncentered_images[frame], cmap="gray")
    axes[0].set_title("Cleaned (Original Position)")

    # RIGHT: overlay
    axes[1].imshow(overlay)

    # polarity vector (yellow)
    if not np.isnan(dy_pol):
        axes[1].arrow(cx, cy, dx_pol * 2, dy_pol * 2,
                      color='yellow', head_width=5)

    # movement vector (green)
    if not np.isnan(dy_mov):
        axes[1].arrow(cx, cy, dx_mov * 4, dy_mov * 4,
                      color='lime', head_width=5)

    # cosine label
    axes[1].text(
        5, 15,
        f"cos={cos_val:.2f}",
        color='white',
        bbox=dict(facecolor='black', alpha=0.6)
    )

    axes[1].set_title(f"Overlay (Frame {frame})")
    axes[1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    # -------------------------
    # Cosine similarity plot
    # -------------------------
    #st.subheader("Cosine similarity over time")
    #st.line_chart(cosine_sim)

if "results" in st.session_state:

    if st.button("Compute Front/Back Lamellipodia"):

        data = st.session_state.results

        lam_masks = data["lam_masks"]
        body_centers = data["body_centers"]
        polarity_vectors = data["polarity_vectors"]

        front_masks, back_masks = split_front_back_lamellipodia(
            lam_masks,
            body_centers,
            polarity_vectors
        )

        # simple area analysis
        front_area = [np.sum(m) for m in front_masks]
        back_area = [np.sum(m) for m in back_masks]

        st.session_state.front_back = {
            "front_masks": front_masks,
            "back_masks": back_masks,
            "front_area": front_area,
            "back_area": back_area
        }
        front_mov, back_mov = split_front_back_lamellipodia_movement(
            lam_masks,
            body_centers,
            movement_vectors
        )

        st.session_state.front_back_movement = {
            "front": front_mov,
            "back": back_mov
        }


        st.success("Front/Back analysis complete!")

if "front_back" in st.session_state:

    st.subheader("Front vs Back Lamellipodia Area")

    fb = st.session_state.front_back
    front_masks = fb["front_masks"]
    back_masks = fb["back_masks"]

    front_area = np.array([np.sum(m) for m in front_masks])
    back_area = np.array([np.sum(m) for m in back_masks])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    ax.plot(front_area, label="Front area", color="green")
    ax.plot(back_area, label="Back area", color="red")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Area (pixels)")
    ax.set_title("Front vs Back Lamellipodia Area")
    ax.legend()

    st.pyplot(fig)

if "front_back" in st.session_state and "front_back_movement" in st.session_state:

    fb_pol = st.session_state.front_back
    fb_mov = st.session_state.front_back_movement

    front_pol = fb_pol["front_masks"][frame]
    front_mov = fb_mov["front"][frame]

    st.subheader("Front Comparison (Polarity vs Movement)")

    # create overlay
    comp_overlay = np.zeros((*img.shape, 3))

    # polarity front → BLUE
    comp_overlay[..., 2] = front_pol.astype(float)

    # movement front → GREEN
    comp_overlay[..., 1] = front_mov.astype(float)

    # overlap → CYAN (blue + green automatically)
    # background intensity (optional faint grayscale)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
    comp_overlay += img_norm[..., None] * 0.2

    fig3, ax3 = plt.subplots()
    ax3.imshow(comp_overlay)

    ax3.set_title("Blue = Polarity Front | Green = Movement Front | Cyan = Overlap")
    ax3.axis("off")

    st.pyplot(fig3)

_='''if "front_back_movement" in st.session_state:

    st.subheader("Leading Edge Area vs Velocity")

    fb_mov = st.session_state.front_back_movement
    front_mov = fb_mov["front"]

    # compute metrics
    velocity = [
        np.sqrt(dy**2 + dx**2) if not np.isnan(dy) else np.nan
        for dy, dx in movement_vectors
    ]

    leading_edge_area = [np.sum(m) for m in front_mov]

    # remove NaNs
    x = np.array(velocity)
    y = np.array(leading_edge_area)

    valid = ~np.isnan(x)
    x = x[valid]
    y = y[valid]

    # scatter plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set_xlabel("Velocity")
    ax.set_ylabel("Leading Edge Area")
    ax.set_title("Leading Edge Area vs Velocity")

    st.pyplot(fig)'''

_ = '''if "results" in st.session_state:

    st.subheader("Leading Edge Velocity vs Cell Velocity")

    lam_masks = data["lam_masks"]
    body_centers = data["body_centers"]
    movement_vectors = data["movement_vectors"]

    # compute tip + velocity
    tips = compute_leading_edge_tip(
        lam_masks,
        body_centers,
        movement_vectors
    )

    tip_velocity = compute_tip_velocity(tips)

    cell_velocity = [
        np.sqrt(dy**2 + dx**2) if not np.isnan(dy) else np.nan
        for dy, dx in movement_vectors
    ]

    # clean NaNs
    x = np.array(cell_velocity)
    y = np.array(tip_velocity)

    valid = ~np.isnan(x) & ~np.isnan(y)

    x = x[valid]
    y = y[valid]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    ax.set_xlabel("Cell Velocity")
    ax.set_ylabel("Leading Edge Velocity")
    ax.set_title("Leading Edge vs Cell Velocity")

    st.pyplot(fig)'''

if "results" in st.session_state:

    st.subheader("Turning Events vs Cosine Similarity")

    movement_vectors = data["movement_vectors"]
    cosine_sim = np.array(data["cosine_sim"])

    turning_events, angles, angle_changes = detect_turning_events_from_movement(
        movement_vectors,
        angle_threshold=np.deg2rad(30)
    )

    turning_events = np.array(turning_events)

    valid = ~np.isnan(cosine_sim)

    x = np.arange(len(cosine_sim))[valid]
    y = cosine_sim[valid]
    t = turning_events[valid]

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # cosine similarity
    ax.plot(x, y, label="Cosine similarity", color="blue")

    # turning events
    ax.scatter(
        x[t == 1],
        y[t == 1],
        color="red",
        label="Turning event",
        zorder=3
    )

    # -------------------------
    # CURRENT FRAME LINE
    # -------------------------
    frame = st.session_state.get("current_frame", 0)

    ax.axvline(
        x=frame,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Current frame"
    )

    ax.set_xlabel("Frame")
    ax.set_ylabel("Cosine similarity")
    ax.set_title("Turning Events vs Cosine Similarity")
    ax.legend()

    st.pyplot(fig)
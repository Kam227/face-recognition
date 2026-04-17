"""
feature_extraction.py
Extracts three complementary feature descriptors from a 64x64 face crop:

  HOG (Histogram of Oriented Gradients)
    Captures the shape and structure of facial features by summarising
    the distribution of edge directions across small image patches.

  LBP (Local Binary Patterns)
    Captures fine texture by comparing each pixel to its neighbours
    and building a histogram of the resulting binary codes.

  Colour histogram (HSV)
    Captures skin tone and overall colour distribution — useful for
    distinguishing individuals under consistent lighting.

HOG and LBP are L2-normalised independently before concatenation so that
the larger HOG vector does not dominate the smaller LBP histogram.
"""

import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
    visualize=True,
    transform_sqrt=True,
)

LBP_SCALES = [(1, 8), (2, 16), (3, 24)] # (radius, n_points) — multi-scale

COLOR_BINS = 8


def extract_features(
    face_crop_64: np.ndarray,
    face_crop_color: np.ndarray | None = None,
) -> dict:
    """
    Inputs:
      face_crop_64    — 64×64 grayscale numpy array
      face_crop_color — 64×64 BGR numpy array (optional)
    Output:
      dict containing feature vectors and visualisation images
    """
    # HOG 
    hog_vector, hog_image = hog(face_crop_64, **HOG_PARAMS)

    # Multi-scale LBP
    lbp_hists = []
    lbp_map = None # keep the finest scale for visualisation
    for radius, n_points in LBP_SCALES:
        n_bins = n_points + 2
        lbp = local_binary_pattern(face_crop_64, n_points, radius, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        lbp_hists.append(hist)
        if lbp_map is None:
            lbp_map = lbp
    lbp_hist = np.concatenate(lbp_hists)
    hog_norm = hog_vector / (np.linalg.norm(hog_vector) + 1e-8)
    lbp_norm = lbp_hist  / (np.linalg.norm(lbp_hist)  + 1e-8)

    # Colour histogram (HSV)
    if face_crop_color is not None and face_crop_color.ndim == 3:
        hsv = cv2.cvtColor(face_crop_color, cv2.COLOR_BGR2HSV)
        fg = (face_crop_color[:, :, 0] > 0) | \
             (face_crop_color[:, :, 1] > 0) | \
             (face_crop_color[:, :, 2] > 0)
        if fg.sum() > 0:
            h_hist = np.histogram(hsv[:, :, 0][fg], bins=COLOR_BINS, range=(0, 180), density=True)[0]
            s_hist = np.histogram(hsv[:, :, 1][fg], bins=COLOR_BINS, range=(0, 256), density=True)[0]
            v_hist = np.histogram(hsv[:, :, 2][fg], bins=COLOR_BINS, range=(0, 256), density=True)[0]
        else:
            h_hist = s_hist = v_hist = np.zeros(COLOR_BINS)
        color_hist = np.concatenate([h_hist, s_hist, v_hist])
    else:
        color_hist = np.zeros(COLOR_BINS * 3)

    descriptor = np.concatenate([hog_norm, lbp_norm, color_hist])

    return {
        "hog_vector": hog_vector,
        "hog_image": hog_image,
        "lbp_vector": lbp_hist,
        "lbp_map": lbp_map,
        "color_hist": color_hist,
        "descriptor": descriptor,
    }

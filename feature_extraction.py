"""
feature_extraction.py
Extracts two complementary feature descriptors from a 64x64 face crop:

  HOG (Histogram of Oriented Gradients)
    Captures the shape and structure of facial features by summarising
    the distribution of edge directions across small image patches.

  LBP (Local Binary Patterns)
    Captures fine texture by comparing each pixel to its neighbours
    and building a histogram of the resulting binary codes.

The two vectors are concatenated into a single descriptor that will
be used by a classifier in a later project stage.
"""

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

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS   = LBP_POINTS + 2   # number of uniform LBP patterns


def extract_features(face_crop_64: np.ndarray) -> dict:
    """
    Input : 64x64 grayscale numpy array
    Output: dict containing feature vectors and visualisation images
    """
    # HOG
    hog_vector, hog_image = hog(face_crop_64, **HOG_PARAMS)

    # LBP
    lbp_map  = local_binary_pattern(face_crop_64, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp_map.ravel(), bins=LBP_BINS, range=(0, LBP_BINS), density=True
    )

    return {
        "hog_vector": hog_vector,   # 1764-dim
        "hog_image":  hog_image,    # for visualisation
        "lbp_vector": lbp_hist,     # 10-dim
        "lbp_map":    lbp_map,      # for visualisation
        "descriptor": np.concatenate([hog_vector, lbp_hist]),  # combined
    }

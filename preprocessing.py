"""
preprocessing.py
Preprocessing steps applied to every face image:
  1. Resize to 128x128
  2. Convert to grayscale
  3. CLAHE  — normalizes uneven lighting across images
  4. Gaussian blur — reduces noise
"""

import cv2
import numpy as np

TARGET_SIZE = (128, 128)


def load_image(path: str) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is None:
        print(f"  [WARNING] Could not load: {path}")
    return img


def preprocess(img_bgr: np.ndarray) -> dict:
    resized   = cv2.resize(img_bgr, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    gray      = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    denoised  = cv2.GaussianBlur(equalized, (3, 3), sigmaX=1.0)

    return {
        "resized":   resized,
        "gray":      gray,
        "equalized": equalized,
        "denoised":  denoised,
    }

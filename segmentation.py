"""
segmentation.py
Face detection and segmentation pipeline:

  1. Haar Cascade face detection — locates the face bounding box
  2. Eye detection + alignment — rotates so the eyes are level
  3. K-means background removal — zeros out non-face pixels within the crop
  4. Centre-crop fallback — used if no face is detected
"""

import cv2
import numpy as np

K = 2 # foreground vs background within the face crop
ITERATIONS = 10
EPSILON = 1.0
ATTEMPTS = 3

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def _align_eyes(gray_crop: np.ndarray, color_crop: np.ndarray):
    """Rotate the face crop so detected eyes are level. Returns (gray, color)."""
    eyes = _eye_cascade.detectMultiScale(
        gray_crop, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
    )
    if len(eyes) < 2:
        return gray_crop, color_crop

    eyes = sorted(eyes, key=lambda e: e[0])[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes[0], eyes[1]
    cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
    cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2

    angle = np.degrees(np.arctan2(cy2 - cy1, cx2 - cx1))
    if abs(angle) > 30:
        return gray_crop, color_crop

    H, W = gray_crop.shape
    center = (float((cx1 + cx2) // 2), float((cy1 + cy2) // 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    gray_aligned = cv2.warpAffine(gray_crop,  M, (W, H), flags=cv2.INTER_LINEAR)
    color_aligned = cv2.warpAffine(color_crop, M, (W, H), flags=cv2.INTER_LINEAR)
    return gray_aligned, color_aligned


def _remove_background(gray_crop: np.ndarray, color_crop: np.ndarray):
    """
    K-means (k=2) within the face crop to separate face from background.
    The cluster containing the centre pixel is kept; the other is zeroed out.
    Returns (masked_gray, masked_color, fg_mask).
    """
    pixels = gray_crop.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ITERATIONS, EPSILON)
    _, labels, _ = cv2.kmeans(
        pixels, K, None, criteria, ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
    )
    labels_2d = labels.reshape(gray_crop.shape)

    cy, cx = gray_crop.shape[0] // 2, gray_crop.shape[1] // 2
    fg_label = int(labels_2d[cy, cx])
    fg_mask = (labels_2d == fg_label).astype(np.uint8)

    gray_masked = gray_crop  * fg_mask
    color_masked = color_crop * fg_mask[:, :, np.newaxis]
    return gray_masked, color_masked, fg_mask


def segment(preprocessed: dict) -> dict:
    gray = preprocessed["denoised"] # 128×128 grayscale
    color = preprocessed["resized"] # 128×128 BGR
    H, W = gray.shape

    # 1. Haar Cascade face detection
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    used_fallback = False
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # largest face
    else:
        used_fallback = True
        w, h = int(W * 0.75), int(H * 0.75)
        x, y = (W - w) // 2, (H - h) // 2

    face_gray = gray [y:y+h, x:x+w].copy()
    face_color = color[y:y+h, x:x+w].copy()

    # 2. Eye detection + alignment
    if not used_fallback:
        face_gray, face_color = _align_eyes(face_gray, face_color)

    # 3. K-means background removal (mask kept for visualization only;
    #    applying it to pre-aligned crops removes discriminative face pixels)
    _, _, fg_mask = _remove_background(face_gray, face_color)

    # 4. Resize to 64×64 (unmasked crops preserve full facial detail)
    face_crop_resized = cv2.resize(face_gray,  (64, 64), interpolation=cv2.INTER_AREA)
    face_crop_resized_color = cv2.resize(face_color, (64, 64), interpolation=cv2.INTER_AREA)

    # Visualisation
    cluster_vis = np.zeros_like(gray)
    fg_mask_full = cv2.resize(fg_mask.astype(np.float32), (w, h)) > 0.5
    cluster_vis[y:y+h, x:x+w][fg_mask_full] = 255
    cluster_vis[y:y+h, x:x+w][~fg_mask_full] = 100

    return {
        "mask": fg_mask,
        "cluster_vis": cluster_vis,
        "primary_box": (x, y, w, h),
        "face_crop": face_gray,
        "face_crop_resized": face_crop_resized,
        "face_crop_resized_color": face_crop_resized_color,
        "used_fallback": used_fallback,
    }

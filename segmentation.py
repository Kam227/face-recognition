"""
segmentation.py
Segments the face region using K-means clustering.

K-means partitions the image pixels into k clusters based on intensity,
separating the face/skin region from the background. The largest
foreground cluster is selected as the face region and cropped.
Falls back to a centre crop if segmentation produces no valid region.
"""

import cv2
import numpy as np

K = 3 # number of clusters (background, mid-tone, skin/face)
ITERATIONS = 10
EPSILON = 1.0
ATTEMPTS = 3


def segment(preprocessed: dict) -> dict:
    gray = preprocessed["denoised"]
    H, W = gray.shape

    pixels = gray.reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, ITERATIONS, EPSILON)
    _, labels, centers = cv2.kmeans(
        pixels, K, None, criteria, ATTEMPTS, cv2.KMEANS_RANDOM_CENTERS
    )

    labels_2d = labels.reshape(H, W)
    face_cluster = int(np.argmax([centers[i][0] for i in range(K)]))
    mask = (labels_2d == face_cluster).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    used_fallback = False
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w < W * 0.2 or h < H * 0.2:
            used_fallback = True
    else:
        used_fallback = True

    if used_fallback:
        w, h = int(W * 0.75), int(H * 0.75)
        x, y = (W - w) // 2, (H - h) // 2

    face_crop = gray[y : y + h, x : x + w]
    face_crop_resized = cv2.resize(face_crop, (64, 64), interpolation=cv2.INTER_AREA)
    cluster_vis = np.zeros_like(gray)
    step = 255 // K
    for i in range(K):
        cluster_vis[labels_2d == i] = step * (i + 1)

    return {
        "mask": mask,
        "cluster_vis": cluster_vis,
        "primary_box": (x, y, w, h),
        "face_crop": face_crop,
        "face_crop_resized": face_crop_resized,
        "used_fallback": used_fallback,
    }

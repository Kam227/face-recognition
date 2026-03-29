"""
visualization.py
Two outputs:
  1. Pipeline figure — 8-panel walkthrough for a single image.
  2. Confusion matrix — saved after training.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path


def draw_boxes(img_bgr: np.ndarray, boxes: list[tuple]) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return out


def save_pipeline_figure(
    img_bgr: np.ndarray,
    preprocessed: dict,
    segmentation: dict,
    features: dict,
    save_path: str | Path,
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Face Recognition Pipeline", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original")

    axes[0, 1].imshow(preprocessed["gray"], cmap="gray")
    axes[0, 1].set_title("2. Grayscale")

    axes[0, 2].imshow(preprocessed["equalized"], cmap="gray")
    axes[0, 2].set_title("3. CLAHE Equalized")

    axes[0, 3].imshow(preprocessed["denoised"], cmap="gray")
    axes[0, 3].set_title("4. Denoised")

    annotated = draw_boxes(preprocessed["resized"], segmentation["boxes"])
    label = "Haar Cascade" + (" (fallback)" if segmentation["used_fallback"] else "")
    axes[1, 0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f"5. Segmentation\n{label}")

    axes[1, 1].imshow(segmentation["face_crop_resized"], cmap="gray")
    axes[1, 1].set_title("6. Face Crop (64×64)")

    axes[1, 2].imshow(features["hog_image"], cmap="gray")
    axes[1, 2].set_title("7. HOG Descriptor")

    axes[1, 3].imshow(features["lbp_map"], cmap="gray")
    axes[1, 3].set_title("8. LBP Map")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pipeline figure → {save_path}")


def save_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None,
    save_path: str | Path,
    max_classes: int = 20,
):
    """Saves a confusion matrix.  Truncates to max_classes for readability."""
    unique = np.unique(np.concatenate([y_test, y_pred]))
    if len(unique) > max_classes:
        # Keep the max_classes most frequent test labels
        counts = {u: (y_test == u).sum() for u in unique}
        top = sorted(counts, key=counts.get, reverse=True)[:max_classes]
        mask = np.isin(y_test, top)
        y_test = y_test[mask]
        y_pred = y_pred[mask]
        if label_names:
            label_names = [label_names[i] for i in top]

    cm = confusion_matrix(y_test, y_pred)
    fig_w = max(8, len(np.unique(y_test)) * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix  → {save_path}")

"""
run.py
CSE 40535 — Project 03: Preprocessing, Segmentation, Feature Extraction

Set the two directory paths below, then run:
    python run.py

The script will:
  - Walk both dataset directories and find all images
  - Run every image through the full pipeline
  - Print a per-image summary to the terminal
  - Save a pipeline visualisation PNG for the first 5 images of each dataset
  - Print a final summary table

Output images are saved to the 'output/' folder.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocessing import load_image, preprocess
from segmentation import segment
from feature_extraction import extract_features

# ── SET YOUR DATASET PATHS HERE ───────────────────────────────────────────────
BIOID_DIR = "/Users/kam/Desktop/CV/BioID-FaceDatabase-V1.2"   # folder containing BioID_0000.pgm etc.
LFW_DIR   = "/Users/kam/Desktop/CV/archive (5)/lfw_funneled"         # folder containing per-person subfolders
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR      = Path("output")
MAX_VIZ_IMAGES  = 5   # how many pipeline figures to save per dataset


# ── Dataset discovery ─────────────────────────────────────────────────────────

def find_images(directory: str, extensions=("*.pgm", "*.jpg", "*.jpeg", "*.png")) -> list[Path]:
    root   = Path(directory)
    images = []
    for ext in extensions:
        images.extend(root.rglob(ext))
    return sorted(images)


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_pipeline_figure(img_bgr, preprocessed, segmentation, features, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Pipeline: {save_path.stem}", fontsize=13, fontweight="bold")

    # Row 1 — preprocessing
    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")

    axes[0, 1].imshow(preprocessed["gray"], cmap="gray")
    axes[0, 1].set_title("2. Grayscale Conversion")

    axes[0, 2].imshow(preprocessed["equalized"], cmap="gray")
    axes[0, 2].set_title("3. Adaptive Histogram\nEqualization (CLAHE)")

    axes[0, 3].imshow(preprocessed["denoised"], cmap="gray")
    axes[0, 3].set_title("4. Gaussian Filter\n(Denoising)")

    # Row 2 — segmentation + features
    fallback_note = " (centre crop fallback)" if segmentation["used_fallback"] else ""
    axes[1, 0].imshow(segmentation["cluster_vis"], cmap="gray")
    axes[1, 0].set_title(f"5. K-means Clustering\n(k=3{fallback_note})")

    axes[1, 1].imshow(segmentation["face_crop_resized"], cmap="gray")
    axes[1, 1].set_title("6. Segmented Face Region")

    axes[1, 2].imshow(features["hog_image"], cmap="gray")
    axes[1, 2].set_title("7. HOG Feature Extraction\n(Histogram of Oriented Gradients)")

    axes[1, 3].imshow(features["lbp_map"], cmap="gray")
    axes[1, 3].set_title("8. LBP Feature Extraction\n(Local Binary Patterns)")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Per-dataset pipeline ──────────────────────────────────────────────────────

def process_dataset(name: str, directory: str):
    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"Path    : {directory}")
    print(f"{'='*60}")

    images = find_images(directory)
    if not images:
        print(f"[ERROR] No images found in {directory}")
        print("        Check that the path is correct and the dataset is downloaded.")
        return

    print(f"Found {len(images)} images\n")

    viz_dir = OUTPUT_DIR / name
    viz_dir.mkdir(parents=True, exist_ok=True)

    n_success      = 0
    n_fallback     = 0
    n_fail         = 0
    viz_saved      = 0

    for i, img_path in enumerate(images):
        img = load_image(str(img_path))
        if img is None:
            n_fail += 1
            continue

        pre  = preprocess(img)
        seg  = segment(pre)
        feat = extract_features(seg["face_crop_resized"])

        if seg["used_fallback"]:
            n_fallback += 1
            detection_str = "no face cluster found (centre crop fallback)"
        else:
            detection_str = f"face region segmented at {seg['primary_box']}"

        print(f"  [{i+1:>4}/{len(images)}] {img_path.name:<30} "
              f"| HOG: {len(feat['hog_vector'])}d  LBP: {len(feat['lbp_vector'])}d  "
              f"| {detection_str}")

        # Save visualisation for first MAX_VIZ_IMAGES images
        if viz_saved < MAX_VIZ_IMAGES:
            save_pipeline_figure(
                img, pre, seg, feat,
                save_path=viz_dir / f"{img_path.stem}_pipeline.png"
            )
            viz_saved += 1

        n_success += 1

    print(f"\n── Summary: {name} ──────────────────────────────")
    print(f"  Processed successfully    : {n_success}")
    print(f"  K-means segmentation ok   : {n_success - n_fallback}")
    print(f"  Used centre crop fallback : {n_fallback}")
    print(f"  Failed to load            : {n_fail}")
    print(f"  Visualisations saved      : {viz_saved}  →  {viz_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_dataset("BioID", BIOID_DIR)
    process_dataset("LFW",   LFW_DIR)
    print(f"\nDone. All output saved to: {OUTPUT_DIR}/\n")

"""
run.py
CSE 40535 — Project 03 & 04 entry point

Set the dataset paths below, then run:

    python run.py pipeline   — preprocessing, segmentation, feature extraction
    python run.py train      — train and evaluate SVM + MLP classifiers
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

from preprocessing import load_image, preprocess
from segmentation import segment
from feature_extraction import extract_features

BIOID_DIR = "/Users/kam/Desktop/CV/BioID-FaceDatabase-V1.2"
LFW_DIR = "/Users/kam/Desktop/CV/archive (5)/lfw_funneled"

OUTPUT_DIR = Path("output")
MAX_VIZ_IMAGES = 5

# Dataset discovery
def find_images(directory: str, extensions=("*.pgm", "*.jpg", "*.jpeg", "*.png")) -> list:
    root = Path(directory)
    images = []
    for ext in extensions:
        images.extend(root.rglob(ext))
    return sorted(images)

# BioID label from filename 
import re

_BIOID_BOUNDARIES = [
    0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
    220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460,
]

def _bioid_label(image_index: int) -> int:
    for i, boundary in enumerate(reversed(_BIOID_BOUNDARIES)):
        if image_index >= boundary:
            return len(_BIOID_BOUNDARIES) - 1 - i
    return 0

def load_bioid(data_dir: str):
    images = find_images(data_dir)
    paths, labels = [], []
    for p in images:
        m = re.search(r"(\d+)", p.stem)
        idx = int(m.group(1)) if m else 0
        paths.append(p)
        labels.append(_bioid_label(idx))
    n_classes = len(set(labels))
    label_names = [f"Subject_{i:02d}" for i in range(n_classes)]
    return paths, labels, label_names

def load_lfw(data_dir: str, min_samples: int = 10):
    root = Path(data_dir)
    person_dirs = [d for d in root.rglob("*") if d.is_dir() and not d.name.startswith(".")]
    label_map = {}
    for d in person_dirs:
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        if len(imgs) >= min_samples:
            label_map[d.name] = imgs
    if not label_map:
        raise FileNotFoundError(f"No LFW person directories with >= {min_samples} images found in: {data_dir}")
    label_names = sorted(label_map.keys())
    paths, labels = [], []
    for label_id, name in enumerate(label_names):
        for img_path in label_map[name]:
            paths.append(img_path)
            labels.append(label_id)
    return paths, labels, label_names

# Pipeline visualisation
def save_pipeline_figure(img_bgr, preprocessed, segmentation, features, save_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Pipeline: {save_path.stem}", fontsize=13, fontweight="bold")

    axes[0, 0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Image")

    axes[0, 1].imshow(preprocessed["gray"], cmap="gray")
    axes[0, 1].set_title("2. Grayscale Conversion")

    axes[0, 2].imshow(preprocessed["equalized"], cmap="gray")
    axes[0, 2].set_title("3. Adaptive Histogram\nEqualization (CLAHE)")

    axes[0, 3].imshow(preprocessed["denoised"], cmap="gray")
    axes[0, 3].set_title("4. Gaussian Filter\n(Denoising)")

    fallback_note = " (centre crop fallback)" if segmentation["used_fallback"] else ""
    axes[1, 0].imshow(segmentation["cluster_vis"], cmap="gray")
    axes[1, 0].set_title(f"5. Haar Cascade Detection\n+ BG Removal{fallback_note}")

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


def run_pipeline(name: str, directory: str):
    print(f"\n{'='*60}")
    print(f"Dataset : {name}")
    print(f"Path    : {directory}")
    print(f"{'='*60}")

    images = find_images(directory)
    if not images:
        print(f"[ERROR] No images found in {directory}")
        return

    print(f"Found {len(images)} images\n")

    viz_dir = OUTPUT_DIR / name
    viz_dir.mkdir(parents=True, exist_ok=True)

    n_success, n_fallback, n_fail, viz_saved = 0, 0, 0, 0

    for i, img_path in enumerate(images):
        img = load_image(str(img_path))
        if img is None:
            n_fail += 1
            continue

        pre = preprocess(img)
        seg = segment(pre)
        feat = extract_features(seg["face_crop_resized"], seg.get("face_crop_resized_color"))

        if seg["used_fallback"]:
            n_fallback += 1
            detection_str = "no face cluster found (centre crop fallback)"
        else:
            detection_str = f"face region segmented at {seg['primary_box']}"

        print(f"  [{i+1:>4}/{len(images)}] {img_path.name:<30} "
              f"| HOG: {len(feat['hog_vector'])}d  LBP: {len(feat['lbp_vector'])}d  "
              f"| {detection_str}")

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


# Entry point

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"

    if mode == "pipeline":
        run_pipeline("BioID", BIOID_DIR)
        run_pipeline("LFW",   LFW_DIR)
        print(f"\nDone. All output saved to: {OUTPUT_DIR}/\n")

    elif mode == "train":
        from train import train_and_evaluate

        print("\nLoading BioID dataset...")
        b_paths, b_labels, b_names = load_bioid(BIOID_DIR)
        train_and_evaluate(b_paths, b_labels, b_names, "BioID", output_dir=str(OUTPUT_DIR))

        print("\nLoading LFW dataset...")
        l_paths, l_labels, l_names = load_lfw(LFW_DIR, min_samples=30)
        train_and_evaluate(l_paths, l_labels, l_names, "LFW", output_dir=str(OUTPUT_DIR))

        print(f"\nDone. All results saved to: {OUTPUT_DIR}/\n")

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python run.py pipeline")
        print("       python run.py train")
"""
run.py
CSE 40535 — Projects 03, 04, 05

Modes:
    python run.py pipeline — preprocessing, segmentation, feature extraction
    python run.py train — train SVM + MLP, save models
    python run.py test — evaluate on held-out test set
    python run.py demo [bioid|lfw] — pick one random test image, show prediction
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import re
import random

from preprocessing import load_image, preprocess
from segmentation import segment
from feature_extraction import extract_features

# SET YOUR DATASET PATHS HERE 
BIOID_DIR = "/Users/kam/Desktop/CV/BioID-FaceDatabase-V1.2"
LFW_DIR = "/Users/kam/Desktop/CV/archive (5)/lfw_funneled"

OUTPUT_DIR = Path("output")
MAX_VIZ_IMAGES = 5
LFW_MIN_SAMPLES = 30


# Dataset loaders

def find_images(directory: str, extensions=("*.pgm", "*.jpg", "*.jpeg", "*.png")) -> list:
    root = Path(directory)
    images = []
    for ext in extensions:
        images.extend(root.rglob(ext))
    return sorted(images)


def load_bioid(data_dir: str):
    """
    Loads BioID with correct subject labels.
    BioID has 23 subjects, each contributing exactly 20 consecutive images.
    Subject 0 = images 0000-0019, Subject 1 = 0020-0039, ..., Subject 22 = 0440-0459.
    Images beyond 0459 are assigned cyclically: (idx // 20) % 23.
    Source: https://www.bioid.com/face-database/
    """
    images = find_images(data_dir)
    if not images:
        raise FileNotFoundError(f"No BioID images found in: {data_dir}")

    paths, labels = [], []
    for p in images:
        m = re.search(r"(\d+)", p.stem)
        idx = int(m.group(1)) if m else 0
        subject_id = (idx // 20) % 23
        paths.append(p)
        labels.append(subject_id)

    label_names = [f"Subject_{i:02d}" for i in range(23)]
    counts = {i: labels.count(i) for i in range(23)}
    print(f"  BioID: {len(paths)} images, {len(set(labels))} subjects")
    for sid, cnt in sorted(counts.items()):
        print(f"    Subject_{sid:02d}: {cnt} images")
    return paths, labels, label_names


def load_lfw(data_dir: str, min_samples: int = LFW_MIN_SAMPLES):
    """
    Loads LFW. Only includes identities with >= min_samples images.
    Labels are derived from the folder name (person's name), not numeric indices.
    """
    root = Path(data_dir)
    person_dirs = [d for d in root.rglob("*") if d.is_dir() and not d.name.startswith(".")]
    label_map = {}
    for d in person_dirs:
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))
        if len(imgs) >= min_samples:
            label_map[d.name] = sorted(imgs)

    if not label_map:
        raise FileNotFoundError(
            f"No LFW directories with >= {min_samples} images found in: {data_dir}"
        )

    label_names = sorted(label_map.keys())
    paths, labels = [], []
    for label_id, name in enumerate(label_names):
        for img_path in label_map[name]:
            paths.append(img_path)
            labels.append(label_id)

    print(f"  LFW: {len(paths)} images, {len(label_names)} identities (>= {min_samples} images each)")
    for name in label_names:
        print(f"    {name}: {len(label_map[name])} images")
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


def run_pipeline(name: str, directory: str):
    print(f"\n{'='*60}")
    print(f"Dataset : {name}  |  Path: {directory}")
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
        feat = extract_features(seg["face_crop_resized"], seg["face_crop_resized_color"])
        if seg["used_fallback"]:
            n_fallback += 1
            detection_str = "centre crop fallback"
        else:
            detection_str = f"segmented at {seg['primary_box']}"
        print(f"  [{i+1:>4}/{len(images)}] {img_path.name:<30} | {detection_str}")
        if viz_saved < MAX_VIZ_IMAGES:
            save_pipeline_figure(img, pre, seg, feat, save_path=viz_dir / f"{img_path.stem}_pipeline.png")
            viz_saved += 1
        n_success += 1
    print(f"\n── Summary: {name} ──")
    print(f"  Processed: {n_success}  |  K-means ok: {n_success - n_fallback}  |  Fallback: {n_fallback}  |  Failed: {n_fail}")
    print(f"  Visualisations saved: {viz_dir}/")


# Demo

def run_demo(dataset_name: str, paths: list, labels: list, label_names: list):
    import pickle
    from sklearn.model_selection import train_test_split

    out = OUTPUT_DIR / dataset_name
    svm_path = out / "svm_model.pkl"
    scaler_p = out / "scaler.pkl"
    pca_p = out / "pca.pkl"

    if not svm_path.exists():
        print("[ERROR] No trained model found. Run 'python run.py train' first.")
        return

    with open(svm_path, "rb") as f: svm = pickle.load(f)
    with open(scaler_p, "rb") as f: scaler = pickle.load(f)
    with open(pca_p, "rb") as f: pca = pickle.load(f)

    _, test_paths, _, test_labels = train_test_split(
        paths, labels, test_size=0.1, stratify=labels, random_state=99
    )

    idx = random.randint(0, len(test_paths) - 1)
    img_path = test_paths[idx]
    true_label = test_labels[idx]

    img = load_image(str(img_path))
    pre = preprocess(img)
    seg = segment(pre)
    feat = extract_features(seg["face_crop_resized"], seg["face_crop_resized_color"])

    desc = scaler.transform(feat["descriptor"].reshape(1, -1))
    desc = pca.transform(desc)
    pred_label = svm.predict(desc)[0]

    true_name = label_names[true_label]
    pred_name = label_names[pred_label]
    correct = pred_label == true_label

    print(f"\n── Demo Result ({dataset_name}) ─────────────────────")
    print(f"  Image     : {img_path.name}")
    print(f"  True      : {true_name}")
    print(f"  Predicted : {pred_name}")
    print(f"  Result    : {'CORRECT' if correct else 'INCORRECT'}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    color = "green" if correct else "red"
    fig.suptitle(
        f"{'CORRECT' if correct else 'INCORRECT'} — True: {true_name}  |  Predicted: {pred_name}",
        fontsize=12, fontweight="bold", color=color
    )
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[1].imshow(seg["face_crop_resized"], cmap="gray")
    axes[1].set_title("Segmented Face")
    axes[2].imshow(feat["hog_image"], cmap="gray")
    axes[2].set_title("HOG Features")
    for ax in axes:
        ax.axis("off")

    out.mkdir(parents=True, exist_ok=True)
    save_path = out / f"demo_{img_path.stem}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved: {save_path}")


# Entry point

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "pipeline"

    if mode == "pipeline":
        run_pipeline("BioID", BIOID_DIR)
        run_pipeline("LFW", LFW_DIR)
        print(f"\nDone. Output saved to: {OUTPUT_DIR}/\n")

    elif mode == "train":
        from train import train_and_evaluate
        print("\nLoading BioID...")
        b_paths, b_labels, b_names = load_bioid(BIOID_DIR)
        train_and_evaluate(b_paths, b_labels, b_names, "BioID", output_dir=str(OUTPUT_DIR))
        print("\nLoading LFW...")
        l_paths, l_labels, l_names = load_lfw(LFW_DIR)
        train_and_evaluate(l_paths, l_labels, l_names, "LFW", output_dir=str(OUTPUT_DIR))
        print(f"\nDone. Results saved to: {OUTPUT_DIR}/\n")

    elif mode == "test":
        from train import evaluate_test_set
        print("\nLoading BioID...")
        b_paths, b_labels, b_names = load_bioid(BIOID_DIR)
        evaluate_test_set(b_paths, b_labels, b_names, "BioID", output_dir=str(OUTPUT_DIR))
        print("\nLoading LFW...")
        l_paths, l_labels, l_names = load_lfw(LFW_DIR)
        evaluate_test_set(l_paths, l_labels, l_names, "LFW", output_dir=str(OUTPUT_DIR))
        print(f"\nDone. Test results saved to: {OUTPUT_DIR}/\n")

    elif mode == "demo":
        dataset = (sys.argv[2] if len(sys.argv) > 2 else "bioid").lower()
        if dataset == "bioid":
            paths, labels, names = load_bioid(BIOID_DIR)
            run_demo("BioID", paths, labels, names)
        else:
            paths, labels, names = load_lfw(LFW_DIR)
            run_demo("LFW", paths, labels, names)

    else:
        print(f"Unknown mode: '{mode}'")
        print("Usage:")
        print("  python run.py pipeline")
        print("  python run.py train")
        print("  python run.py test")
        print("  python run.py demo [bioid|lfw]")
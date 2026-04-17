"""
train.py
CSE 40535 — Project 04: Classification

Builds a feature matrix from BioID and LFW datasets using the existing
preprocessing → segmentation → feature extraction pipeline, then trains
and evaluates two classifiers:

  1. SVM  (RBF kernel)         — traditional ML baseline
  2. MLP  (fully connected NN) — neural network comparison

Evaluation is done on an 80/20 stratified train/validation split.
Results and confusion matrix PNGs are saved to the output/ directory.
"""

import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)

from preprocessing import load_image, preprocess
from segmentation import segment
from feature_extraction import extract_features


# Feature matrix builder

def build_feature_matrix(image_paths: list, labels: list, dataset_name: str) -> tuple:
    """
    Runs the full pipeline on every image and returns (X, y).
    Skips images that cannot be loaded.
    """
    X, y = [], []
    n = len(image_paths)

    for i, (path, label) in enumerate(zip(image_paths, labels)):
        if i % 100 == 0 or i == n - 1:
            print(f"  [{dataset_name}] Extracting features: {i+1}/{n}", end="\r")

        img = load_image(str(path))
        if img is None:
            continue

        pre = preprocess(img)
        seg = segment(pre)
        feat = extract_features(seg["face_crop_resized"], seg.get("face_crop_resized_color"))

        X.append(feat["descriptor"])
        y.append(label)

    print()
    return np.array(X), np.array(y)


# Plotting helpers

def save_confusion_matrix(y_test, y_pred, label_names, save_path, title, max_classes=20):
    unique = np.unique(np.concatenate([y_test, y_pred]))

    # Truncate to most frequent classes for readability
    if len(unique) > max_classes:
        counts = {u: int((y_test == u).sum()) for u in unique}
        top = sorted(counts, key=counts.get, reverse=True)[:max_classes]
        mask = np.isin(y_test, top)
        y_test = y_test[mask]
        y_pred = y_pred[mask]
        if label_names:
            label_names = [label_names[i] for i in top]
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        display_labels = [label_names[i] if i < len(label_names) else str(i) for i in unique_labels] if label_names else unique_labels
        fig_w = max(8, len(unique_labels) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix saved → {save_path}")


def save_comparison_bar(results: dict, save_path: Path):
    """Bar chart comparing SVM vs MLP accuracy on train and validation sets."""
    labels = list(results.keys())
    train_acc = [results[k]["train_accuracy"] for k in labels]
    val_acc = [results[k]["val_accuracy"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_acc, width, label="Train", color="#4C72B0")
    ax.bar(x + width/2, val_acc, width, label="Validation", color="#DD8452")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train vs Validation Accuracy: SVM vs MLP")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()
    for i, (tr, va) in enumerate(zip(train_acc, val_acc)):
        ax.text(i - width/2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, va + 0.01, f"{va:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison chart saved → {save_path}")


# Main training function

def train_and_evaluate(
    image_paths: list,
    labels: list,
    label_names: list,
    dataset_name: str,
    output_dir: str = "output",
):
    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}  ({len(image_paths)} images, {len(label_names)} classes)")
    print(f"{'='*60}")

    # Build feature matrix
    X, y = build_feature_matrix(image_paths, labels, dataset_name)
    print(f"  Feature matrix: {X.shape}  |  Labels: {y.shape}")

    # Train / validation split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train)}  |  Validation: {len(X_val)}")

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    pca = PCA(n_components=0.99, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    print(f"  PCA: {X_train.shape[1]} components (95 % variance)")

    results = {}

    # 1. SVM
    print(f"\n  Training SVM (RBF kernel)...")
    svm = SVC(kernel="rbf", C=50, gamma="scale", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)

    svm_train_pred = svm.predict(X_train)
    svm_val_pred = svm.predict(X_val)
    svm_train_acc = accuracy_score(y_train, svm_train_pred)
    svm_val_acc = accuracy_score(y_val, svm_val_pred)

    print(f"  SVM  — Train accuracy: {svm_train_acc:.4f}  |  Validation accuracy: {svm_val_acc:.4f}")
    print(f"\n  SVM Validation Report:\n")
    print(classification_report(y_val, svm_val_pred, target_names=label_names, zero_division=0))

    save_confusion_matrix(
        y_val, svm_val_pred, label_names,
        save_path=out / "svm_confusion_matrix.png",
        title=f"SVM Confusion Matrix — {dataset_name} (Validation)"
    )

    results["SVM"] = {
        "train_accuracy": svm_train_acc,
        "val_accuracy": svm_val_acc,
        "val_pred": svm_val_pred,
    }

    # 2. MLP
    print(f"\n  Training MLP (neural network)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    mlp.fit(X_train, y_train)

    mlp_train_pred = mlp.predict(X_train)
    mlp_val_pred = mlp.predict(X_val)
    mlp_train_acc = accuracy_score(y_train, mlp_train_pred)
    mlp_val_acc = accuracy_score(y_val, mlp_val_pred)

    print(f"  MLP  — Train accuracy: {mlp_train_acc:.4f}  |  Validation accuracy: {mlp_val_acc:.4f}")
    print(f"\n  MLP Validation Report:\n")
    print(classification_report(y_val, mlp_val_pred, target_names=label_names, zero_division=0))

    save_confusion_matrix(
        y_val, mlp_val_pred, label_names,
        save_path=out / "mlp_confusion_matrix.png",
        title=f"MLP Confusion Matrix — {dataset_name} (Validation)"
    )

    results["MLP"] = {
        "train_accuracy": mlp_train_acc,
        "val_accuracy": mlp_val_acc,
        "val_pred": mlp_val_pred,
    }

    # Comparison chart
    save_comparison_bar(results, out / "svm_vs_mlp_accuracy.png")

    # Save models
    with open(out / "svm_model.pkl", "wb") as f: pickle.dump(svm, f)
    with open(out / "mlp_model.pkl", "wb") as f: pickle.dump(mlp, f)
    with open(out / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(out / "pca.pkl", "wb") as f: pickle.dump(pca, f)
    print(f"\n  Models saved to {out}/")

    return results
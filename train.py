"""
train.py
CSE 40535 — Project 04 & 05: Classification and Testing

Data split:
  70% train — model is fit on this
  20% val — used to report validation accuracy during development
  10% test — held out entirely, only used in Project 05

Classifiers:
  1. SVM (RBF kernel) — classical ML baseline
  2. MLP — neural network comparison
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
)

from preprocessing import load_image, preprocess
from segmentation import segment
from feature_extraction import extract_features


# Feature matrix 

def build_feature_matrix(image_paths: list, labels: list, dataset_name: str) -> tuple:
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
        feat = extract_features(seg["face_crop_resized"], seg["face_crop_resized_color"])
        X.append(feat["descriptor"])
        y.append(label)
    print()
    return np.array(X), np.array(y)


# Confusion matrix

def save_confusion_matrix(y_true, y_pred, label_names, save_path, title, max_classes=23):
    unique = np.unique(np.concatenate([y_true, y_pred]))
    if len(unique) > max_classes:
        counts = {u: int((y_true == u).sum()) for u in unique}
        top = sorted(sorted(counts, key=counts.get, reverse=True)[:max_classes])
        mask = np.isin(y_true, top)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        unique = np.array(top)

    display_labels = [label_names[i] for i in unique if i < len(label_names)]
    cm = confusion_matrix(y_true, y_pred, labels=unique)
    fig_w = max(8, len(unique) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.85))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def save_comparison_bar(results: dict, save_path: Path):
    labels = list(results.keys())
    train_acc = [results[k]["train_accuracy"] for k in labels]
    val_acc = [results[k]["val_accuracy"] for k in labels]
    x, width = np.arange(len(labels)), 0.35
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
    print(f"  Saved: {save_path}")


# Train

def train_and_evaluate(image_paths, labels, label_names, dataset_name, output_dir="output"):
    out = Path(output_dir) / dataset_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset_name}  ({len(image_paths)} images, {len(label_names)} classes)")
    print(f"{'='*60}")

    X, y = build_feature_matrix(image_paths, labels, dataset_name)
    print(f"  Feature matrix: {X.shape}")

    # Split: 70 train / 20 val / 10 test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=99
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.222, stratify=y_tv, random_state=42
    )
    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)} (held out)")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=0.99, random_state=42)
    X_train = pca.fit_transform(X_train)
    X_val = pca.transform(X_val)
    X_test = pca.transform(X_test)
    print(f"  PCA components: {X_train.shape[1]}")

    results = {}

    # SVM
    print(f"\n  Training SVM...")
    svm = SVC(kernel="rbf", C=50, gamma="scale", class_weight="balanced", random_state=42)
    svm.fit(X_train, y_train)
    svm_train_acc = accuracy_score(y_train, svm.predict(X_train))
    svm_val_acc = accuracy_score(y_val, svm.predict(X_val))
    print(f"  SVM  Train: {svm_train_acc:.4f}  |  Val: {svm_val_acc:.4f}")
    print(classification_report(y_val, svm.predict(X_val), target_names=label_names, zero_division=0))
    save_confusion_matrix(y_val, svm.predict(X_val), label_names,
                          out / "svm_confusion_matrix.png",
                          f"SVM Validation — {dataset_name}")
    results["SVM"] = {"train_accuracy": svm_train_acc, "val_accuracy": svm_val_acc}

    # MLP
    print(f"\n  Training MLP...")
    mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu",
                        max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1, verbose=False)
    mlp.fit(X_train, y_train)
    mlp_train_acc = accuracy_score(y_train, mlp.predict(X_train))
    mlp_val_acc = accuracy_score(y_val, mlp.predict(X_val))
    print(f"  MLP  Train: {mlp_train_acc:.4f}  |  Val: {mlp_val_acc:.4f}")
    print(classification_report(y_val, mlp.predict(X_val), target_names=label_names, zero_division=0))
    save_confusion_matrix(y_val, mlp.predict(X_val), label_names,
                          out / "mlp_confusion_matrix.png",
                          f"MLP Validation — {dataset_name}")
    results["MLP"] = {"train_accuracy": mlp_train_acc, "val_accuracy": mlp_val_acc}

    save_comparison_bar(results, out / "svm_vs_mlp_accuracy.png")

    with open(out / "svm_model.pkl", "wb") as f: pickle.dump(svm, f)
    with open(out / "mlp_model.pkl", "wb") as f: pickle.dump(mlp, f)
    with open(out / "scaler.pkl", "wb") as f: pickle.dump(scaler, f)
    with open(out / "pca.pkl", "wb") as f: pickle.dump(pca, f)
    print(f"\n  Models saved to {out}/")
    return results

# Test
def evaluate_test_set(image_paths, labels, label_names, dataset_name, output_dir="output"):
    out = Path(output_dir) / dataset_name

    print(f"\n{'='*60}")
    print(f"Test Evaluation : {dataset_name}")
    print(f"{'='*60}")

    with open(out / "svm_model.pkl", "rb") as f: svm = pickle.load(f)
    with open(out / "mlp_model.pkl", "rb") as f: mlp = pickle.load(f)
    with open(out / "scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open(out / "pca.pkl", "rb") as f: pca = pickle.load(f)

    X, y = build_feature_matrix(image_paths, labels, dataset_name)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=99
    )
    print(f"  Test set size: {len(X_test)}")
    X_test = pca.transform(scaler.transform(X_test))

    svm_pred = svm.predict(X_test)
    svm_test_acc = accuracy_score(y_test, svm_pred)
    print(f"\n  SVM Test Accuracy: {svm_test_acc:.4f}")
    print(classification_report(y_test, svm_pred, target_names=label_names, zero_division=0))
    save_confusion_matrix(y_test, svm_pred, label_names,
                          out / "svm_test_confusion_matrix.png",
                          f"SVM Test Set — {dataset_name}")

    mlp_pred = mlp.predict(X_test)
    mlp_test_acc = accuracy_score(y_test, mlp_pred)
    print(f"\n  MLP Test Accuracy: {mlp_test_acc:.4f}")
    print(classification_report(y_test, mlp_pred, target_names=label_names, zero_division=0))
    save_confusion_matrix(y_test, mlp_pred, label_names,
                          out / "mlp_test_confusion_matrix.png",
                          f"MLP Test Set — {dataset_name}")

    _save_test_comparison(
        {"SVM": svm_test_acc, "MLP": mlp_test_acc},
        out / "test_accuracy_comparison.png",
        dataset_name
    )
    return {"SVM": svm_test_acc, "MLP": mlp_test_acc}


def _save_test_comparison(test_acc: dict, save_path: Path, dataset_name: str):
    labels = list(test_acc.keys())
    t_vals = [test_acc[k] for k in labels]
    x, width = np.arange(len(labels)), 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, t_vals, width, color="#55A868")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Test Set Accuracy — {dataset_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    for i, t in enumerate(t_vals):
        ax.text(i, t + 0.01, f"{t:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
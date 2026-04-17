# CSE 40535 — Project 03 & 04
## Face Recognition Pipeline: Preprocessing, Segmentation, Feature Extraction, and Classification

---

## Overview

This project builds a complete face recognition pipeline using two publicly available datasets: the **BioID Face Database** and **Labeled Faces in the Wild (LFW)**. The pipeline covers four stages: data preprocessing, face detection and segmentation, feature extraction, and classification. The goal is to take raw face images and transform them into structured numerical descriptors, then use those descriptors to train and evaluate two classifiers — a Support Vector Machine (SVM) and a Multi-Layer Perceptron (MLP).

---

## Datasets

- **BioID Face Database** — 1,521 grayscale images of 24 subjects captured under controlled but varied lighting and pose conditions. Downloaded from the [official BioID website](https://www.bioid.com/face-database/).
- **Labeled Faces in the Wild (LFW)** — ~13,000 color images of public figures collected from the web, organized by identity. Downloaded from [Kaggle](https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download). Only identities with 30 or more images are used for classification.

---

## Methods

### 1. Data Preprocessing

Every image from both datasets is passed through the same four-step preprocessing pipeline before any further processing takes place.

**Step 1 — Resize to 128×128**
Images across both datasets come in different resolutions. Resizing them all to a fixed 128×128 ensures that subsequent operations are consistent and comparable regardless of the original image dimensions.

**Step 2 — Grayscale Conversion**
Converting to grayscale reduces each image from a 3-channel representation to a single intensity channel, which simplifies computation and focuses the pipeline on structural properties of the face.

**Step 3 — Adaptive Histogram Equalization (CLAHE)**
CLAHE normalizes local contrast across the image, making it more robust to lighting differences between datasets. Unlike standard histogram equalization which operates globally, CLAHE works on small local regions, preserving important facial detail rather than blowing out highlights or crushing shadows.

**Step 4 — Gaussian Filter (Denoising)**
A small Gaussian blur (3×3 kernel, σ=1.0) is applied to reduce high-frequency noise, particularly relevant for the older BioID images which show more sensor noise.

---

### 2. Face Detection and Segmentation

After preprocessing, the pipeline isolates the face region in three steps.

**Step 1 — Haar Cascade Face Detection**
A pre-trained Haar Cascade detector (`haarcascade_frontalface_default.xml`) locates the face bounding box in the image. This replaces the earlier K-means approach, which partitioned pixel intensities globally and was unreliable on varied real-world backgrounds. If no face is detected, the pipeline falls back to a centre crop covering 75% of the image.

**Step 2 — Eye Detection and Alignment**
Within the detected face crop, a second Haar Cascade (`haarcascade_eye.xml`) locates the eyes. If two eyes are found, the image is rotated so they are level — a standard normalization step in face recognition that removes head-tilt variation. Rotations greater than 30° are treated as detection errors and skipped.

**Step 3 — K-means Background Removal**
K-means clustering (k=2) is applied within the detected face crop to separate the face from remaining background (hair, clothing edges, shadows). The cluster containing the centre pixel of the crop is treated as foreground; the other cluster is zeroed out. This ensures that extracted features are driven by face appearance rather than background content.

The masked face crop is then resized to 64×64 pixels for feature extraction.

---

### 3. Feature Extraction

Three complementary descriptors are extracted from each 64×64 face crop and concatenated into a single feature vector.

**HOG — Histogram of Oriented Gradients**
HOG describes the shape and structure of a face by computing the distribution of edge directions in small local patches. The image is divided into 8×8 pixel cells; within each cell a 9-bin histogram of gradient orientations is computed, then normalized across 2×2 blocks of cells. The result is a 1,764-dimensional vector capturing facial geometry — jawline shape, eye position, nose structure — in a way that is relatively robust to small lighting changes.

**Multi-Scale LBP — Local Binary Patterns**
LBP captures fine texture by comparing each pixel to its neighbors and building a histogram of the resulting binary codes. To capture texture at multiple scales, histograms are computed at three (radius, n_points) configurations — (1, 8), (2, 16), and (3, 24) — and concatenated into a 42-dimensional vector. This captures both fine micro-texture and coarser structural patterns.

**Colour Histogram (HSV)**
An 8-bin histogram is computed for each of the three HSV channels (hue, saturation, value) using only foreground pixels, producing a 24-dimensional colour descriptor. This captures skin tone and overall colour distribution that can differentiate individuals under consistent lighting conditions.

**Per-Descriptor L2 Normalisation**
Before concatenation, the HOG vector and the LBP vector are each independently L2-normalised. Without this step, the 1,764-dimensional HOG vector would dominate the distance metric, effectively making LBP and colour features irrelevant. The final descriptor is approximately 1,830 dimensions.

---

## Illustrations

The following pipeline figures show how each image is processed through all stages, from raw input to final feature maps. Results are shown for both datasets.

### BioID Dataset

**BioID_0000**
![BioID_0000 Pipeline](output/BioID/BioID_0000_pipeline.png)

**BioID_0001**
![BioID_0001 Pipeline](output/BioID/BioID_0001_pipeline.png)

**BioID_0002**
![BioID_0002 Pipeline](output/BioID/BioID_0002_pipeline.png)

---

### LFW Dataset

**Aaron Eckhart**
![Aaron Eckhart Pipeline](output/LFW/Aaron_Eckhart_0001_pipeline.png)

**Aaron Guiel**
![Aaron Guiel Pipeline](output/LFW/Aaron_Guiel_0001_pipeline.png)

**AJ Cook**
![AJ Cook Pipeline](output/LFW/AJ_Cook_0001_pipeline.png)

---

## Running the Code

### Setup

```bash
conda env create -f environment.yml
conda activate cv-project03
```

### Configuration

Open `run.py` and set the dataset paths at the top of the file:

```python
BIOID_DIR = "/path/to/BioID_database"
LFW_DIR   = "/path/to/lfw_funneled"
```

### Run

```bash
# Pipeline visualizations only
python run.py pipeline

# Train and evaluate classifiers
python run.py train
```

Pipeline visualizations for the first 5 images of each dataset are saved to `output/BioID/` and `output/LFW/`. Classification results, confusion matrices, and saved models are written to the same directories.

---

## Part 4 — Classification Report

### Overview

This section covers the classification stage of the pipeline. Two classifiers are trained and evaluated on the feature vectors produced by the pipeline described above: a **Support Vector Machine (SVM)** with an RBF kernel and a **Multi-Layer Perceptron (MLP)**. Both are evaluated on an 80/20 stratified train/validation split.

---

### Experimental Setup

**Feature preprocessing for classification:**
After feature extraction, the 1,830-dimensional descriptors are standardized using `StandardScaler` (zero mean, unit variance). Principal Component Analysis (PCA) is then applied, retaining enough components to explain 99% of the variance. This step is critical: with approximately 22 training samples per class and nearly 1,800 features, classifiers overfit severely without dimensionality reduction. PCA compresses correlated features and forces the model to generalize across a lower-dimensional subspace.

**Classifiers:**
- **SVM** — RBF kernel, C=50, `gamma="scale"`, `class_weight="balanced"`. The balanced class weight compensates for the significant imbalance between classes.
- **MLP** — Two hidden layers (512, 256 units), ReLU activation, early stopping on an internal 10% validation split, up to 300 epochs.

**LFW filtering:**
Only identities with at least 30 images are included in the LFW experiment. This yields a more tractable problem: fewer classes but more training samples per class, which improves the signal-to-noise ratio for the classifier without changing the underlying pipeline.

---

### Results

#### BioID Dataset (24 classes, 1,521 images)

| Model | Train Accuracy | Validation Accuracy |
|-------|:--------------:|:-------------------:|
| SVM   | 1.00           | **0.89**            |
| MLP   | 0.99           | **0.88**            |

![BioID Accuracy Chart](output/BioID/svm_vs_mlp_accuracy.png)

#### LFW Dataset (filtered to ≥30 images/class)

| Model | Train Accuracy | Validation Accuracy |
|-------|:--------------:|:-------------------:|
| SVM   | 1.00           | **0.64**            |
| MLP   | 0.97           | **0.65**            |

![LFW Accuracy Chart](output/LFW/svm_vs_mlp_accuracy.png)

---

### Confusion Matrix Analysis

#### BioID

![BioID SVM Confusion Matrix](output/BioID/svm_confusion_matrix.png)
![BioID MLP Confusion Matrix](output/BioID/mlp_confusion_matrix.png)

The BioID confusion matrices reveal a significant class imbalance: Subject_23 accounts for the majority of images in the dataset (images indexed from 460 onward), while all other subjects have only 4 validation samples each. As a result, accuracy on this dataset is partially inflated — the classifier correctly handles the dominant class and performs moderately on the remaining minority subjects. Errors among subjects 0–22 are scattered rather than concentrated, suggesting the classifier is struggling with the small per-class sample count rather than with systematic confusion between particular identities.

Both SVM and MLP perform nearly identically on BioID, which is expected — the dataset is small and relatively clean (controlled lighting, fixed background), so the feature vectors are informative enough that both models converge to similar decision boundaries.

#### LFW

![LFW SVM Confusion Matrix](output/LFW/svm_confusion_matrix.png)
![LFW MLP Confusion Matrix](output/LFW/mlp_confusion_matrix.png)

The LFW confusion matrices tell a more nuanced story. The most-represented identities — Jean Chrétien, Ariel Sharon, George W. Bush, and Junichiro Koizumi — are classified with high recall, as reflected by the bright diagonal entries for those rows. Minority classes with fewer validation samples show much weaker performance and are often misclassified as one of the dominant identities.

This is a well-known failure mode for multi-class classifiers trained on imbalanced data: the model learns to default toward high-frequency classes when uncertain. The `class_weight="balanced"` setting on the SVM partially mitigates this, but cannot fully overcome the underlying data scarcity for minority subjects.

---

### Discussion

**BioID vs. LFW performance gap.**
BioID achieves ~89% validation accuracy while LFW reaches ~65%. The gap reflects fundamental differences in dataset difficulty. BioID is a controlled lab dataset with consistent backgrounds, lighting, and camera distance — the pipeline's Haar Cascade detector works reliably and the resulting face crops are clean and consistent. LFW, by contrast, consists of web-collected photographs with extreme variation in pose, lighting, resolution, and occlusion. Haar Cascade detection is less reliable on these images, which means some face crops are misaligned or noisy before features are even extracted.

**Overfitting.**
Both classifiers achieve near-perfect training accuracy (97–100%) while validation accuracy is substantially lower. This is a classic overfitting signature and is largely attributable to the high dimensionality of the raw feature space relative to the number of training samples per class. PCA significantly reduced this gap compared to training without it — before adding PCA, LFW SVM validation accuracy was approximately 16%. The residual gap reflects that even with PCA, the number of examples per identity (roughly 24 for training after the split) is insufficient to fully characterize the variation within each class.

**SVM vs. MLP.**
The two classifiers perform nearly identically across both datasets, with differences of at most 1 percentage point. This parity suggests that the bottleneck is the feature representation itself, not the classifier capacity. More expressive models (e.g., deeper networks) would not meaningfully improve results given the same hand-crafted feature vectors — the information ceiling is set by what HOG, LBP, and colour histograms can capture, not by the classifier's ability to separate them.

**Pipeline improvements and their impact.**
The most impactful change across the project was replacing intensity-based K-means segmentation with Haar Cascade face detection. The original K-means approach would frequently select the wrong region (bright backgrounds, overexposed patches) and produce noisy face crops, degrading all downstream features. Haar Cascade detection provides geometrically consistent crops, which is the prerequisite for any face-specific feature to be meaningful. Secondary improvements — eye alignment, background removal within the crop, multi-scale LBP, and per-descriptor L2 normalization — each contributed incrementally to the final accuracy.

**Limitations and future directions.**
The 35% error rate remaining on LFW is a fundamental ceiling for hand-crafted HOG+LBP features on a real-world multi-class recognition problem. Classical methods (Eigenfaces, Fisherfaces) achieved similar performance ranges on comparable benchmarks in the pre-deep-learning era. The path to substantially higher accuracy would require learned features — convolutional neural networks routinely exceed 99% on LFW by learning to extract identity-discriminative representations directly from pixel data, rather than relying on manually designed descriptors. Within the classical framework, the most tractable remaining improvement would be face landmark detection for more precise geometric alignment (e.g., aligning eye centres to fixed pixel coordinates), which would make HOG features more geometrically consistent across subjects.

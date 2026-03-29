# CSE 40535 — Project 03
## Face Recognition Pipeline: Preprocessing, Segmentation, and Feature Extraction

---

## Overview

This project builds a face recognition pipeline using two publicly available datasets: the **BioID Face Database** and **Labeled Faces in the Wild (LFW)**. This update covers the first stage of the pipeline — data preprocessing, face segmentation, and feature extraction. The goal is to take raw face images and transform them into structured numerical descriptors that can eventually be used to train a face recognition classifier.

---

## Datasets

- **BioID Face Database** — 1,521 grayscale images of 23 subjects captured under controlled but varied lighting and pose conditions. Downloaded from the [official BioID website](https://www.bioid.com/face-database/).
- **Labeled Faces in the Wild (LFW)** — ~13,000 color images of public figures collected from the web, organized by identity. Downloaded from [Kaggle](https://www.kaggle.com/datasets/atulanandjha/lfwpeople?resource=download). Only identities with 10 or more images are used.

---

## Methods

### 1. Data Preprocessing

Every image from both datasets is passed through the same four-step preprocessing pipeline before any further processing takes place.

**Step 1 — Resize to 128×128**
Images across both datasets come in different resolutions. Resizing them all to a fixed 128×128 ensures that subsequent operations are consistent and comparable regardless of the original image dimensions.

**Step 2 — Grayscale Conversion**
Color information is not particularly useful for face recognition based on shape and texture. Converting to grayscale reduces each image from a 3-channel representation to a single intensity channel, which simplifies computation and focuses the pipeline on the structural properties of the face.

**Step 3 — Adaptive Histogram Equalization (CLAHE)**
The two datasets were captured under very different conditions — BioID images are taken in a controlled lab setting, while LFW images are real-world photographs with varying lighting. CLAHE (Contrast Limited Adaptive Histogram Equalization) normalizes local contrast across the image, making it more robust to these lighting differences. Unlike standard histogram equalization which operates globally, CLAHE works on small local regions, which helps preserve important facial detail rather than blowing out highlights or crushing shadows.

**Step 4 — Gaussian Filter (Denoising)**
A small Gaussian blur (3×3 kernel, σ=1.0) is applied to reduce high-frequency noise. This is especially relevant for the BioID images, which are older and show more sensor noise than the LFW images.

---

### 2. Segmentation

After preprocessing, the goal is to isolate the face region from the rest of the image. This is done using **K-means clustering** with k=3.

K-means partitions the pixel intensities of the grayscale image into three clusters. In most face images, these clusters roughly correspond to dark regions (hair, background shadows), mid-tone regions (clothing, some background), and bright regions (skin, highlights). The cluster with the highest mean intensity is selected as the candidate face region. The bounding box of the largest connected component within that cluster is then used to crop the face.

If the detected region is too small (less than 20% of the image in either dimension), the segmentation falls back to a simple centre crop covering 75% of the image. This handles edge cases like images with very dark or unusual lighting where the clustering does not produce a clean face region.

The face crop is then resized to 64×64 pixels and passed into feature extraction.

---

### 3. Feature Extraction

Two complementary feature descriptors are extracted from each face crop.

**HOG — Histogram of Oriented Gradients**
HOG describes the shape and structure of a face by computing the distribution of edge directions in small local patches across the image. The image is divided into 8×8 pixel cells, and within each cell a histogram of gradient orientations (9 bins) is computed. These histograms are then normalized across 2×2 blocks of cells. The result is a 1,764-dimensional feature vector that captures the overall facial geometry — things like the shape of the jawline, the position of the eyes, and the structure of the nose — in a way that is relatively robust to small changes in lighting.

**LBP — Local Binary Patterns**
LBP captures fine texture information by comparing each pixel to its surrounding neighbors. For each pixel, the neighbors are thresholded against the center pixel value, producing a binary code. A histogram of these codes over the entire image gives a compact 10-dimensional descriptor of the micro-texture of the face — things like skin texture, edge sharpness, and local contrast patterns that differ between individuals.

These two vectors are concatenated into a single combined descriptor for each image, which will be passed into a classifier in a future project stage.

---

## Why These Methods?

The choice of HOG and LBP as feature extractors is motivated by the fact that they are both well-established in the face recognition literature and map directly onto concepts covered in class. HOG captures the large-scale geometric structure of a face (gradients, edges, oriented shapes), while LBP captures the fine-scale texture. Together they provide complementary information about both the shape and surface appearance of a face.

CLAHE was chosen over standard histogram equalization because the two datasets have very different lighting conditions. Standard equalization would apply the same transformation globally, which can produce harsh artifacts. CLAHE handles this more gracefully by working locally.

K-means was chosen for segmentation because it is a straightforward unsupervised method for partitioning pixels by intensity, which is well-suited to separating a face (typically the brightest region in a controlled shot) from the background. It does not require any additional model or pre-trained data.

---

## Illustrations

The following pipeline figures show how each image is processed through all stages, from the raw input to the final feature maps. Results are shown for both datasets.

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
python run.py
```

Pipeline visualizations for the first 5 images of each dataset are saved to `output/BioID/` and `output/LFW/`. A summary of how many images were processed and how many produced a successful segmentation is printed to the terminal.

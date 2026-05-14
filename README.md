# 🌿 Deforestation Segmentation using Deep Multimodal Learning

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.x-D00000?logo=keras&logoColor=white)](https://keras.io/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Binary pixel-level segmentation of deforested regions from Sentinel-2 satellite imagery using a U-Net architecture with EfficientNet-B3 encoder.**

Based on the research methodology from [arXiv:2307.04916](https://arxiv.org/abs/2307.04916).

---

## 📖 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [References](#references)

---

## 🎯 Overview

This project implements a **deep learning pipeline** for detecting deforestation from satellite imagery. Using Sentinel-2 multispectral data from the Amazon rainforest, the model performs binary segmentation to classify each pixel as either **forested** (0) or **deforested** (1).

### Key Features

| Feature | Description |
|---------|-------------|
| **Architecture** | U-Net with EfficientNet-B3 encoder (ImageNet pretrained) |
| **Input** | 4-band satellite imagery (RGB + NIR) at 256×256 resolution |
| **Output** | Binary deforestation mask with pixel-level precision |
| **Loss Function** | Combined Dice Loss + Binary Focal Loss |
| **Framework** | TensorFlow 2.x / Keras (no external segmentation library!) |

### Target Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Pixel Accuracy | ≥ 88% | Overall classification accuracy |
| F1 Score (Dice) | ≥ 0.85 | Harmonic mean of precision and recall |
| IoU (Jaccard) | ≥ 0.75 | Intersection over Union |

---

## 🏗️ Architecture

The model follows a **U-Net encoder-decoder** architecture with skip connections:



**Why EfficientNet-B3?** It provides an excellent accuracy-efficiency trade-off, uses compound scaling for balanced depth/width/resolution, and ImageNet pretraining gives strong feature extraction even on satellite imagery.

---

## 📊 Dataset

### MultiEarth 2023

The [MultiEarth 2023](https://sites.google.com/view/rainforest-challenge/multiearth-2023) dataset from the CVPR 2023 workshop contains:

| Property | Details |
|----------|---------|
| **Source** | Sentinel-2 L2A satellite imagery |
| **Region** | Amazon rainforest |
| **Bands Used** | B2 (Blue), B3 (Green), B4 (Red), B8 (NIR) |
| **Masks** | Binary — 1 = deforested, 0 = forest |
| **Format** | GeoTIFF files |

### Download

```bash
# Option 1: Using azcopy (recommended)
python -m src.download_data

# Option 2: Generate synthetic data for testing
python -m src.download_data --generate_synthetic --num_samples 500
```

---

## 📈 Results

<!-- Update this section after training with real data -->

| Metric | Value |
|--------|-------|
| **Pixel Accuracy** | TBD |
| **F1 Score** | TBD |
| **IoU** | TBD |
| **Training Time** | TBD |

### Sample Predictions

<!-- Add prediction visualizations here after training -->
*Run `python -m src.predict` to generate prediction visualizations in `results/predictions/`*

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deforestation-segmentation.git
cd deforestation-segmentation

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Download / Generate Data

```bash
# Generate synthetic data for testing
python -m src.download_data --generate_synthetic --num_samples 500

# Or download real MultiEarth 2023 data (requires azcopy)
python -m src.download_data
```

### 2. Train the Model

```bash
# Train with synthetic data (CPU-friendly settings)
python -m src.train --epochs 15 --batch_size 4

# Train with real data
python -m src.train --data_dir data/multiearth --epochs 30

# With frozen encoder (faster convergence)
python -m src.train --freeze_encoder --epochs 20
```

### 3. Run Inference

```bash
# Predict on test set
python -m src.predict

# With custom model and data
python -m src.predict --model results/checkpoints/best_model.keras --data_dir data/multiearth
```

### 4. Launch Demo App

```bash
python app.py
# Opens at http://localhost:7860
```

---

## 📁 Project Structure

```
deforestation-segmentation/
├── data/
│   └── multiearth/              # Satellite GeoTIFF files + masks
├── notebooks/
│   └── EDA.ipynb                # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── dataset.py               # tf.data pipeline + augmentation
│   ├── model.py                 # U-Net + EfficientNet-B3 (custom)
│   ├── losses.py                # Dice + Focal loss functions
│   ├── metrics.py               # IoU, F1, Pixel Accuracy
│   ├── train.py                 # model.fit() + callbacks
│   ├── predict.py               # Inference + visualization
│   ├── download_data.py         # Dataset download helper
│   └── utils.py                 # Plot helpers, seed, utilities
├── results/
│   ├── checkpoints/             # Saved model weights
│   ├── logs/                    # Training logs (CSV)
│   └── predictions/             # Output mask images
├── app.py                       # Gradio demo web app
├── requirements.txt
└── README.md
```

---

## 🔬 Technical Details

### Loss Function: Combined Dice + Focal Loss

```
L_total = L_dice + L_focal

L_dice  = 1 - (2 × |A ∩ B| + ε) / (|A| + |B| + ε)
L_focal = -α(1-p_t)^γ × log(p_t)    [α=0.25, γ=2.0]
```

- **Dice Loss** handles the class imbalance problem (deforested areas are often small)
- **Focal Loss** focuses training on hard-to-classify boundary pixels

### Data Pipeline

```
GeoTIFF → rasterio → Band Selection (4 bands) → Normalization
→ albumentations (flip, rotate, noise, dropout) → tf.data (batch + prefetch)
```

### Training Configuration (CPU-Optimized)

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Optimizer | AdamW (lr=1e-4, wd=1e-5) |
| LR Schedule | Cosine Decay |
| Epochs | 15 |
| Early Stopping | Patience 5 |
| Dropout | 0.3 (Spatial) |

---

## 📚 References

1. **Paper:** [Deep Multimodal Learning for Deforestation Detection](https://arxiv.org/abs/2307.04916)
2. **Dataset:** [MultiEarth 2023 — CVPR Workshop](https://sites.google.com/view/rainforest-challenge/multiearth-2023)
3. **EfficientNet:** [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
4. **U-Net:** [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)

---

## 📄 License

This project is for educational and research purposes. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built with ❤️ using TensorFlow & Keras
</p>

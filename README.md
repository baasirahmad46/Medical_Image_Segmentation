# 🧠 Brain Tumor Segmentation using U-Net

> Automated brain tumor detection and segmentation from MRI scans using the U-Net deep learning architecture, trained and evaluated on the BRISC 2025 dataset.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10.0-orange)
![Dice Score](https://img.shields.io/badge/Test%20Dice%20Score-0.8280-brightgreen)
![Accuracy](https://img.shields.io/badge/Accuracy-98.96%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This project presents a complete end-to-end pipeline for automated brain tumor segmentation from contrast-enhanced T1-weighted MRI scans. It uses the **U-Net convolutional neural network architecture** — the gold standard for medical image segmentation — trained on the **BRISC 2025** dataset comprising 6,000 expert-annotated MRI images.

The system identifies and delineates tumor regions at the pixel level, classifying each pixel as either **tumor** or **background**, achieving a **Test Dice Score of 0.8280** on completely unseen test images.

A **Streamlit web application** is included for real-time clinical demonstration — upload any brain MRI scan and get automated tumor segmentation results instantly.

---

## 🎯 Results

| Metric | Score |
|---|---|
| **Test Dice Score** | **0.8280** |
| **Test IoU (Jaccard)** | **0.7412** |
| **Test Pixel Accuracy** | **98.96%** |
| **False Negative Rate** | **0.21%** |
| Excellent predictions (Dice ≥ 0.90) | 496 / 860 images (57.7%) |
| Good predictions (Dice ≥ 0.75) | 671 / 860 images (78.0%) |

---

## 📁 Project Structure

```
Medical_Image_Segmentation/
│
├── notebooks/
│   ├── 1_exploration_EDA.ipynb       # Dataset analysis & integrity check
│   ├── 2_preprocessing.ipynb         # Image preprocessing pipeline
│   ├── 3_model_building.ipynb        # U-Net architecture design
│   ├── 4_training.ipynb              # Model training with auto-resumption
│   └── 5_evaluation.ipynb            # Comprehensive evaluation
│
├── models/
│   ├── best_model.keras              # Best model (by validation Dice)
│   ├── checkpoint.keras              # Latest epoch checkpoint
│   └── history.npy                   # Training history
│
├── data/
│   ├── X.npy                         # Preprocessed training images
│   ├── Y.npy                         # Preprocessed training masks
│   ├── X_test.npy                    # Preprocessed test images
│   └── Y_test.npy                    # Preprocessed test masks
│
├── outputs/
│   ├── training_curves.png           # Loss, Dice, IoU curves
│   ├── confusion_matrix.png          # Pixel-wise confusion matrix
│   ├── dice_distribution.png         # Per-image Dice distribution
│   ├── best_predictions.png          # Top 8 predictions
│   ├── worst_predictions.png         # Challenging cases
│   └── overlay_visualization.png     # Tumor overlays
│
├── app.py                            # Streamlit web application
├── run_app.bat                       # One-click Windows launcher
└── README.md
```

---

## 🗂️ Dataset

**BRISC 2025** — Brain tumor Image Segmentation and Classification 2025

| Property | Details |
|---|---|
| Total Images | 6,000 MRI scans |
| Training Set | 3,933 images |
| Test Set | 860 images |
| Image Resolution | 512 × 512 px (original) → 256 × 256 px (processed) |
| Tumor Categories | Glioma, Meningioma, Pituitary Tumor, No Tumor |
| Annotators | Certified radiologists and physicians |
| Source | [Kaggle — BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025) |

> ⚠️ **Integrity Note:** During our EDA, we identified 98 cross-set duplicate images (11.4% of the test set, all Meningioma category) present in both train and test folders under different filenames. This is a dataset quality issue in BRISC 2025 itself. Our test metrics may be marginally optimistic; 88.6% of test images (762/860) are confirmed completely unseen.

---

## 🏗️ Model Architecture

**U-Net** with a reduced filter configuration adapted for 4GB GPU hardware:

```
Input (256×256×3)
      │
 ┌────▼────┐
 │Encoder  │  [32 → 64 → 128 → 256] filters + max-pooling
 └────┬────┘
      │
 ┌────▼────┐
 │Bottleneck│  512 filters (16×16 spatial resolution)
 └────┬────┘
      │
 ┌────▼────┐
 │Decoder  │  [256 → 128 → 64 → 32] filters + UpSampling
 │+ Skip   │  Skip connections from encoder to decoder
 │  Conns  │
 └────┬────┘
      │
 Output (256×256×1) — Sigmoid activation
```

| Parameter | Standard U-Net | This Project |
|---|---|---|
| Filter progression | [64,128,256,512,1024] | [32,64,128,256,512] |
| Total parameters | ~31.4 million | ~7.8 million |
| Reason for reduction | — | 4GB GPU VRAM constraint |

**Loss Function:** Combined Binary Cross Entropy + Dice Loss  
**Optimizer:** Adam (lr = 0.0001)

---

## ⚙️ Setup and Installation

### Prerequisites

- Windows 10/11 (for native GPU support with TF 2.10)
- NVIDIA GPU with CUDA support (4GB+ VRAM recommended)
- [Anaconda](https://www.anaconda.com/download) installed

### Step 1 — Create Conda Environment

```bash
conda create -n tf210 python=3.10
conda activate tf210
```

### Step 2 — Install CUDA and cuDNN

```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```

### Step 3 — Install Python Packages

```bash
pip install tensorflow==2.10.0
pip install numpy==1.23.5
pip install opencv-python==4.8.0.76
pip install matplotlib scikit-learn seaborn jupyter
pip install streamlit pillow
```

### Step 4 — Verify GPU Detection

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# Should print: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Step 5 — Download Dataset

Download the BRISC 2025 dataset from [Kaggle](https://www.kaggle.com/datasets/briscdataset/brisc2025) and place it in:

```
data/brisc2025/segmentation_task/
├── train/
│   ├── images/   ← 3,933 .jpg files
│   └── masks/    ← 3,933 .png files
└── test/
    ├── images/   ← 860 .jpg files
    └── masks/    ← 860 .png files
```

---

## 🚀 Running the Project

### Step 1 — Data Exploration
Open and run `notebooks/1_exploration_EDA.ipynb`

### Step 2 — Preprocessing
Open and run `notebooks/2_preprocessing.ipynb`  
*(Generates X.npy, Y.npy, X_test.npy, Y_test.npy in the data/ folder)*

### Step 3 — Model Building
Open and run `notebooks/3_model_building.ipynb`  
*(Reviews the full U-Net architecture — 31.4M parameters standard design)*

### Step 4 — Training

```bash
conda activate tf210
jupyter notebook
```

Open `notebooks/4_training.ipynb` and run all cells.

> **Multi-session training:** Training automatically resumes from the last checkpoint if interrupted. Just re-run the notebook — it detects and loads the existing checkpoint.

### Step 5 — Evaluation
Open and run `notebooks/5_evaluation.ipynb`

---

## 🌐 Streamlit Web Application

### Launch the App

**Option A — Double-click `run_app.bat`** (Windows)

**Option B — Terminal:**

```bash
conda activate tf210
cd path/to/Medical_Image_Segmentation
python -m streamlit run app.py
```

App opens at `http://localhost:8501`

### How to Use

1. Upload any brain MRI scan (JPG or PNG)
2. The model automatically preprocesses and segments the image
3. View results: Original MRI | Predicted Mask | Tumor Overlay (Red)
4. See metrics: Tumor Detected (YES/NO), Tumor Area %, Model Confidence

---

## 📊 Training Details

| Parameter | Value |
|---|---|
| Epochs trained | ~40 (across multiple sessions) |
| Batch size | 4 (reduced to 2 for OOM prevention) |
| Best validation Dice | 0.8448 (epoch 30) |
| GPU | NVIDIA GTX 1650 (4GB, 3GB limit set) |
| Time per epoch | ~17–30 minutes |
| Total training time | ~15–20 hours |

### Training Callbacks

| Callback | Purpose |
|---|---|
| ModelCheckpoint (best) | Saves model when val Dice improves |
| ModelCheckpoint (every epoch) | Enables seamless training resumption |
| ReduceLROnPlateau | Halves LR after 5 epochs of no improvement |
| EarlyStopping | Stops after 15 epochs of no improvement |
| SaveHistoryCallback (custom) | Saves metrics after every epoch — crash-safe |

---

## 🔬 Evaluation Results

### Confusion Matrix (Pixel-level across 860 test images)

| | Predicted Background | Predicted Tumor |
|---|---|---|
| **Actual Background** | 55,244,802 TN (98.02%) | 220,740 FP (0.39%) |
| **Actual Tumor** | 119,358 FN (0.21%) | 776,060 TP (1.38%) |

> The **False Negative rate of 0.21%** is the most critical clinical metric — our model misses very few actual tumor pixels, which is essential for safe clinical use.

### Per-Image Performance

| Category | Images | Percentage |
|---|---|---|
| Excellent (Dice ≥ 0.90) | 496 | 57.7% |
| Good (Dice ≥ 0.75) | 671 | 78.0% |
| Fair (0.50 ≤ Dice < 0.75) | 102 | 11.9% |
| Poor (Dice < 0.50) | 87 | 10.1% |

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core language |
| TensorFlow | 2.10.0 | Deep learning framework |
| Keras | 2.10.0 | Model building API |
| NumPy | 1.23.5 | Array operations |
| OpenCV | 4.8.0.76 | Image processing |
| Matplotlib | Latest | Visualization |
| scikit-learn | Latest | Data splitting, metrics |
| Seaborn | Latest | Confusion matrix plot |
| Streamlit | 1.58.0 | Web application |
| CUDA | 11.2 | GPU acceleration |
| cuDNN | 8.1 | Deep learning GPU ops |

---

## 📖 Reference

This project is methodologically guided by:

> **Medical Image Segmentation using Deep Learning**  
> *Software Impacts, Volume 14, Elsevier, 2022*  
> DOI: [10.1016/j.simpa.2022.100429](https://www.sciencedirect.com/science/article/pii/S2772442522000429)

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Made with ❤️ and a lot of GPU hours 🖥️*

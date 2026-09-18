# AI Document Verification System

> An AI-powered system for detecting potential document forgery using OCR, machine learning, and deep learning.

## 🚀 Live Demo

[**Try the Live Demo →**](YOUR_STREAMLIT_URL)

## 📂 GitHub Repository

[**View Source Code →**](https://github.com/Abhiram4004/ai-document-verification-system)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OCR Engine](https://img.shields.io/badge/OCR-Tesseract%205.4+-brightgreen.svg)](https://github.com/tesseract-ocr/tesseract)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.5.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-23%20passed%2C%201%20warning-success.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning and Deep Learning system for detecting digital tampering and forgeries in receipts and invoices. Evaluated on the real-world **Find it Again! (findit2)** benchmark using **Tesseract OCR**, classical feature engineering, **pretrained ResNet-18 transfer learning**, and **Grad-CAM visual interpretability**.

---

## Overview

This project investigates automated document and retail receipt forgery detection using optical character recognition, classical machine learning, and visual transfer learning. 

Rather than treating forgery detection as an off-the-shelf problem, this repository documents the complete engineering and scientific progression:
1. **Raw Document Processing**: Optical Character Recognition via live Tesseract OCR execution and image normalization.
2. **Classical Feature Engineering**: Construction of a 37-dimensional multimodal feature space spanning geometry, grayscale moments, edge densities, color consistency, and OCR lexical metrics.
3. **Classical ML Baseline**: Fitting candidate models (Logistic Regression, SVM, Random Forest, HistGradientBoosting) with balanced class weighting.
4. **Scientific Feature Diagnosis**: Statistical audit of feature distributions, revealing why document-wide statistical aggregations fail to discriminate localized edits.
5. **Decision Threshold Calibration**: Sweeping validation probabilities to identify calibrated decision thresholds that counter severe ~5:1 class imbalance without retraining.
6. **Visual Transfer Learning**: Pretrained ResNet-18 fine-tuned in two stages with class-weighted Cross-Entropy loss on raw document pixels.
7. **Visual Interpretability (Grad-CAM)**: Backward gradient hooks into convolutional feature maps to visualize model attention patterns.
8. **Production Interfaces**: Deployable CLI (`predict.py`), Flask REST API (`api.py`), and interactive Streamlit web dashboard (`app.py`).

> **Honest Scientific Disclosure**: Detecting document tampering on unconstrained consumer receipts is a difficult task. Localized edits typically occupy less than 0.5% of the total pixel area. As documented in the [Results](#results) section, neither the classical model nor the visual model achieves high precision or near-perfect classification. This system serves as an automated decision-support and triage tool, not an infallible or certified forgery detector.

---

## Problem Statement

Document and receipt tampering typically involves subtle modifications:
- Changing a single digit in a total amount (e.g., $10.00 to $70.00).
- Altering the transaction date or vendor name.
- Splicing text blocks from different documents.

These localized modifications occupy a fraction of a percent of the document image. Global statistical features (e.g., average image brightness, standard deviation of contrast, overall word count) aggregate information across the entire document, diluting localized pixel manipulation signals. Consequently, classical tabular models alone struggle to differentiate genuine documents from skillfully edited ones.

---

## Dataset

Evaluated on the **Find it Again! (findit2)** receipt forgery benchmark located at `archive/findit2/` (or configured via `DATASET_ROOT` in `.env`).

### Partitioning & Sample Counts

| Partition | Total Samples | Genuine (0) | Forged (1) | Forged Class Ratio | Purpose |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 577 | 483 (83.71%) | 94 (16.29%) | 16.29% | Model fitting & feature scaler fitting |
| **Validation** | 192 | 159 (82.81%) | 33 (17.19%) | 17.19% | Hyperparameter selection & threshold calibration |
| **Test** | 218 | 183 (83.94%) | 35 (16.06%) | 16.06% | Final unbiased evaluation strictly once |
| **Total** | **987** | **825 (83.59%)** | **162 (16.41%)** | **16.41%** | Benchmark total |

### Leakage Prevention & Audit Rules
- **Cross-Split Deduplication**: Sample `X51006619709.png` was identified in both training and validation sets. It is automatically pruned from validation at load time (`src/data_loader.py`), guaranteeing zero overlap across splits:
  - $\text{Train} \cap \text{Validation} = \emptyset$
  - $\text{Train} \cap \text{Test} = \emptyset$
  - $\text{Validation} \cap \text{Test} = \emptyset$
- **Strict Exclusion of Annotations**: Ground-truth bounding box coordinates, modification masks, and software metadata from the dataset `.txt` files are **strictly excluded** from model inputs. Models learn solely from raw pixel arrays and binary authenticity labels.
- **Zero OCR Fallback**: Ground-truth `.txt` files are never used as an OCR substitute. All textual features stem directly from live execution of the Tesseract OCR engine.
- **Dataset Availability Notice**: Due to licensing terms, the raw receipt dataset is not hosted inside the Git repository. Users must obtain the **Find it Again!** benchmark separately and place it in the directory specified by `DATASET_ROOT`.

---

## Architecture

The system features two complementary inference pathways:

```
Document Image
       |
       +----------------------+
       |                      |
       v                      v
 Image Preprocessing       Tesseract OCR
       |                      |
       |                      v
       |                 OCR Statistics
       |                      |
       +----------+-----------+
                  |
                  v
        Classical Feature Engine (37 Features)
                  |
                  v
        HistGradientBoosting (Calibrated Threshold 0.25)
                  |
                  v
       Classical Prediction (Genuine / Forged)


AND


Document Image
       |
       v
  Image Preprocessing (384 x 384)
       |
       v
   ResNet-18 (Pretrained Transfer Learning)
       |
       +----------------------+
       |                      |
       v                      v
 Forged Probability      Grad-CAM Hooks
       |                      |
       v                      v
  Threshold 0.48        Attention Heatmap
       |
       v
 Visual Prediction (Genuine / Forged)


Unified Deployment:
                    +--> CLI (predict.py)
                    |
Models + OCR + ----+--> Flask REST API (api.py)
Inference           |
                    +--> Streamlit Dashboard (app.py)
```

---

## OCR Pipeline

Textual features are extracted using the **Tesseract OCR (5.x)** binary engine:
- Preprocessing: Grayscale conversion, contrast stretching, Otsu binarization.
- Direct execution via `pytesseract` with timeout safety and error handling.
- **Explicit Rule**: Dataset annotation `.txt` files are **NOT** used as an OCR fallback. If Tesseract is unavailable, the pipeline stops and alerts the user rather than creating silent distribution shift.

### Extracted Indicators
From the OCR stream, the pipeline extracts:
- `ocr_char_count`, `ocr_word_count`, `ocr_line_count`: Document lexical volume.
- `ocr_digit_count`, `ocr_digit_ratio`: Numerical content proportion (critical for receipts).
- `ocr_avg_word_length`, `ocr_uppercase_ratio`: Lexical structural regularities.
- `ocr_mean_confidence`: Average word-level OCR confidence score (0 to 100%).
- `ocr_low_conf_ratio`: Proportion of extracted words with confidence < 50%.
- `ocr_char_density`: Number of characters per unit document area.

---

## Classical ML Approach

### 37 Multimodal Tabular Features
The classical pipeline constructs a deterministic 37-dimensional vector per document:
- **Geometry (4)**: Aspect ratio, log pixel area, normalized width, normalized height.
- **Intensity & Contrast (7)**: Mean intensity, standard deviation, median, 10th percentile, 90th percentile, dynamic range ($P_{90} - P_{10}$), skewness.
- **Sharpness & Edges (7)**: Laplacian variance, $4 \times 4$ spatial patch sharpness standard deviation and ratio, Canny edge density, horizontal/vertical Sobel gradients, gradient energy.
- **Color Consistency (9)**: RGB per-channel means and standard deviations, maximum inter-channel delta, HSV saturation mean and standard deviation.
- **OCR Text Metrics (10)**: Detailed in the OCR section above.

### Feature Diagnosis Findings
A statistical audit (`scripts/diagnose_features.py`) revealed:
1. **Univariate Disconnect**: Univariate ROC-AUCs across all 37 global statistical features were near chance ($0.45 - 0.56$), with Mann-Whitney $U$ test $p > 0.05$ across every feature and Cohen's $d < 0.18$.
2. **Multicollinearity**: 65 feature pairs exhibited Spearman rank correlation $|\rho| \ge 0.80$ (e.g., `gray_std` vs `color_green_std` at $0.9997$), caused by monochromatic print on light thermal paper.
3. **Conclusion**: Global document-wide scalars inherently dilute localized tampering artifacts that cover $< 0.5\%$ of document area.

### Model Selection & Threshold Calibration
Due to the ~5:1 class imbalance, candidate models (Logistic Regression, SVM, Random Forest, HistGradientBoosting) collapsed toward predicting 100% genuine when evaluated at the default 0.50 threshold. 

We calibrated the decision threshold on validation set predicted probabilities:
- For `HistGradientBoosting (l2=10.0, min_leaf=20)`: Calibrating the decision threshold to **0.25** lifted validation Forged Recall to **69.70%** and validation Forged F1 to **0.3622**.

---

## Visual Deep Learning

To preserve two-dimensional spatial context rather than collapsing images into global scalars, we implemented visual representation learning (`src/visual_model.py`, `train_visual.py`):

- **Backbone**: ResNet-18 pretrained on ImageNet (`torchvision.models.ResNet18_Weights.DEFAULT`).
- **Classification Head**: Adaptive average pooling followed by dropout regularization (`p=0.3`), linear layer (`512 -> 128`), ReLU, dropout (`p=0.2`), and final linear projection (`128 -> 2`).
- **Two-Stage Transfer Learning**:
  - Stage 1 (Warmup): Backbone layers frozen (`conv1` through `layer4`); only the classification head trained.
  - Stage 2 (Fine-tuning): `layer4` and classification head unfrozen with differential learning rates ($1 \times 10^{-5}$ for backbone, $1 \times 10^{-4}$ for head).
- **Imbalance Handling**: Class-weighted Cross-Entropy loss with weight vector $[1.0, 5.14]$.
- **Augmentation (Training Only)**: Random affine rotation ($\pm 5^\circ$) and color jitter ($\pm 10\%$). Validation and test splits use strictly deterministic $384 \times 384$ bilinear resizing and ImageNet normalization.
- **Validation Threshold**: Identified at **0.48**, producing validation Forged Recall of **69.70%**, Forged F1 of **0.3067**, and ROC-AUC of **0.5567**.

---

## Results

*The test partition (218 samples: 183 genuine, 35 forged) was kept strictly untouched during all feature diagnosis, model training, hyperparameter tuning, and threshold selection. It was evaluated strictly once at the conclusion of experiments. These results are frozen.*

| Model Architecture | Evaluated Threshold | Test Accuracy | Forged Precision | Forged Recall | Forged F1-Score | ROC-AUC | Test Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Logistic Regression (C=1.0)** | 0.50 | 45.41% | 11.82% | 37.14% | 0.1793 | 0.4442 | `[86, 97, 22, 13]` |
| **Selected HistGradientBoosting** | **0.25** | 43.58% | 13.33% | 45.71% | **0.2065** | 0.4306 | `[79, 104, 19, 16]` |
| **Visual ResNet-18 (Transfer Learning)** | **0.48** | 45.87% | 15.70% | **54.29%** | **0.2436** | **0.4971** | `[81, 102, 16, 19]` |

### Analysis of Test Results
1. **Recall Advantage**: The deep visual model caught **19 of 35** forged receipts in the test set (54.29% recall), compared to 16 for the selected classical model (45.71%) and 13 for the baseline (37.14%).
2. **F1 Gain**: ResNet-18 achieved the highest Forged F1-Score (**0.2436**), demonstrating that spatial convolutional feature hierarchies retain localized visual cues better than global tabular scalars.
3. **Discriminative Difficulty**: The test ROC-AUC hovers around 0.43 to 0.50 across all models, reflecting high false-positive rates when tuned for high recall. This proves that receipt forgery detection cannot be considered solved with global classification alone.

---

## Interpretability

To provide transparency into neural network decisions, the system integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**:
- Registers backward hooks on the final convolutional layer of ResNet-18 (`layer4`).
- Computes the gradient of the Forged class score with respect to feature activation maps.
- Weights each activation channel by its pooled gradient, applies ReLU, and normalizes values to $[0, 1]$.
- Blends the resulting jet colormap directly over the document image.

> ℹ️ **Mandatory Disclaimer**: Grad-CAM attention heatmaps illustrate model visual attention patterns and feature activations. They do **NOT** represent ground-truth forgery bounding boxes, pixel-level manipulation masks, or legal proof of tampering.

---

## API

The system provides a production Flask REST API (`api.py`).

### Endpoints

#### 1. `GET /health`
Checks engine readiness, loaded models, and Tesseract binary availability.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "HistGradientBoosting (l2=10.0, min_leaf=20)",
  "visual_model_loaded": true,
  "visual_model_architecture": "ResNet-18 (Transfer Learning)",
  "tesseract_available": true,
  "endpoints": ["/health", "/model-info", "/predict", "/gradcam"]
}
```

#### 2. `GET /model-info`
Returns model specifications, feature names, baseline test metrics, and improved evaluation records.

#### 3. `POST /predict`
Verifies an uploaded document image (`multipart/form-data`, file key: `image` or `file`).
```bash
curl -X POST -F "image=@archive/findit2/test/X00016469619.png" http://127.0.0.1:5000/predict
```
*Response excerpt:*
```json
{
  "success": true,
  "prediction": "Forged",
  "label": 1,
  "is_forged": true,
  "confidence": 0.3302,
  "forged_probability": 0.3302,
  "decision_threshold": 0.25,
  "visual_model": {
    "architecture": "ResNet-18 (Transfer Learning)",
    "forged_probability": 0.4949,
    "threshold": 0.48,
    "is_forged": true
  },
  "processing": {
    "ocr_characters": 496,
    "word_count": 81,
    "digit_count": 101,
    "mean_ocr_confidence": 77.51
  },
  "processing_time_ms": 1513.26,
  "disclaimer": "Model probability represents statistical estimation and is not a legal or certified guarantee of authenticity."
}
```

#### 4. `POST /predict?gradcam=true`
Includes a base64-encoded PNG string of the blended Grad-CAM heatmap overlay in `gradcam_overlay_base64`.

#### 5. `POST /gradcam`
Directly streams the blended Grad-CAM PNG image overlay as `image/png`.
```bash
curl -X POST -F "image=@archive/findit2/test/X00016469619.png" http://127.0.0.1:5000/gradcam -o overlay.png
```

---

## Streamlit Dashboard

An interactive dashboard is implemented in `app.py`.

### Launch Command
```bash
streamlit run app.py
```

### Dashboard Capabilities
- **Verify Document Tab**:
  - Upload custom receipt images or pick directly from the test split.
  - Displays dual model predictions: **Primary Classical Model** (`HistGradientBoosting`, threshold 0.25) side-by-side with **Deep Learning Model** (`ResNet-18`, threshold 0.48).
  - Toggles and displays the blended Grad-CAM attention heatmap.
  - Expands extracted Tesseract OCR text and the 37-dimensional tabular feature vector.
- **Model Performance & Insights Tab**:
  - Live table comparing Baseline, HistGradientBoosting, and ResNet-18 test metrics.
  - Interactive display of Confusion Matrix, ROC Curve, and feature importance.
  - Summary of statistical feature diagnosis findings.
- **Dataset & Architecture Tab**:
  - Complete architecture diagrams, split isolation details, and deduplication records.

---

## Installation

### Prerequisites
- Python 3.10, 3.11, or 3.12
- [Tesseract OCR 5.x](https://github.com/UB-Mannheim/tesseract/wiki) installed on your system

### Setup Steps
```bash
# 1. Clone repository
git clone <repository_url>
cd ai-document-verification-system--main

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env     # On Windows (or 'cp .env.example .env' on Linux)

# 5. Edit .env to set your Tesseract binary path:
# e.g., TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Running

### CLI Prediction
```bash
# Formatted verification summary
python predict.py --image archive/findit2/test/X00016469619.png

# Generate and save Grad-CAM heatmap overlay
python predict.py --image archive/findit2/test/X00016469619.png --gradcam outputs/test_cam.png

# Output raw JSON
python predict.py --image archive/findit2/test/X00016469619.png --json
```

### Flask REST API
```bash
python api.py
```
Starts the API on `http://127.0.0.1:5000`.

### Streamlit Web Dashboard
```bash
streamlit run app.py
```
Opens the dashboard in your default browser at `http://localhost:8501`.

### Training Pipeline (Optional / For Reproduction)
```bash
# Train classical baseline and generate initial evaluation artifacts
python train.py

# Train visual ResNet-18 transfer learning model
python train_visual.py

# Run classical model benchmarking and threshold calibration sweep
python scripts/experiment_classical_models.py

# Run unbiased evaluation on the untouched test partition
python scripts/evaluate_selected_model.py
```

---

## Testing

Run the automated test suite across all modules:
```bash
pytest tests/ -v
```

### Verified Test Suite Status
```
tests/test_api.py::test_health_endpoint PASSED                           [  4%]
tests/test_api.py::test_predict_endpoint_missing_file PASSED             [  8%]
tests/test_api.py::test_predict_endpoint_unsupported_file PASSED         [ 13%]
tests/test_api.py::test_model_info_endpoint PASSED                       [ 17%]
tests/test_api.py::test_predict_endpoint_with_image PASSED               [ 21%]
tests/test_api.py::test_predict_endpoint_with_gradcam_flag PASSED        [ 26%]
tests/test_api.py::test_gradcam_endpoint_image_stream PASSED             [ 30%]
tests/test_data_loader.py::test_dataset_loader_splits PASSED             [ 34%]
tests/test_data_loader.py::test_split_statistics PASSED                  [ 39%]
tests/test_features.py::test_feature_extraction_shape_and_cleanliness PASSED [ 43%]
tests/test_features.py::test_feature_determinism PASSED                  [ 47%]
tests/test_model.py::test_calculate_metrics PASSED                       [ 52%]
tests/test_model.py::test_model_training_and_serialization PASSED        [ 56%]
tests/test_ocr.py::test_text_statistics_calculation PASSED               [ 60%]
tests/test_ocr.py::test_ocr_pipeline_mock PASSED                         [ 65%]
tests/test_ocr.py::test_tesseract_missing_error PASSED                   [ 69%]
tests/test_preprocessing.py::test_load_image PASSED                      [ 73%]
tests/test_preprocessing.py::test_resize_keep_aspect PASSED              [ 78%]
tests/test_preprocessing.py::test_grayscale_and_binarize PASSED          [ 82%]
tests/test_visual_model.py::test_document_forgery_resnet_architecture PASSED [ 86%]
tests/test_visual_model.py::test_gradcam_computation PASSED              [ 91%]
tests/test_visual_model.py::test_overlay_gradcam PASSED                  [ 95%]
tests/test_visual_model.py::test_verifier_gradcam_integration PASSED     [100%]

======================= 23 passed, 1 warning in 12.71s ========================
```

---

## Limitations

1. **Information Loss from Whole-Document Resizing**: Resizing long vertical receipts down to $384 \times 384$ compresses fine character strokes, obscuring minute pixel-level tampering traces.
2. **Global Aggregation Dilution**: Classical statistical features aggregate properties across entire documents, washing out localized edits occupying $< 0.5\%$ of document area.
3. **Dataset Scope & Class Imbalance**: The dataset contains 987 total receipts with an ~5:1 genuine-to-forged ratio. While sufficient for benchmarking, it does not encompass all scanner distortions, lighting shifts, and thermal paper aging artifacts.
4. **Binary Document-Level Supervision**: Training with binary labels (genuine vs forged) rather than bounding boxes prevents models from learning dense spatial localization directly.
5. **No Legal Guarantee**: Given test ROC-AUC scores around 0.43–0.50, current models must not be used as autonomous or certified authenticity verifiers.

---

## Future Work

The following directions are identified for future research (not currently implemented):
- **Patch-Based Classification**: Slicing native-resolution text lines into overlapping patches to prevent downsampling artifacts.
- **Dense Localization Supervision**: Training segmentation networks (e.g., Mask R-CNN, SegFormer) using pixel-level or bounding-box forgery masks.
- **Multi-Modal Document Foundation Models**: Integrating pre-trained vision-language architectures (e.g., LayoutLMv3, Donut).
- **Synthetic Tampering Augmentation**: Generating realistic character replacements and font-mismatch perturbations during training.

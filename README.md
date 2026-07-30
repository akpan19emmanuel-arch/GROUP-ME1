# ME1 — Rotten Mango vs Formalin-Mixed Mango Classifier

## Overview

This project trains a binary image classifier to distinguish **Rotten** mangoes from
**Formalin-Mixed** mangoes using **transfer learning** with MobileNetV2 pre-trained on
ImageNet. The model is built with TensorFlow / Keras and follows a two-phase training
strategy: a feature-extraction phase (frozen base) followed by a fine-tuning phase
(top 30 layers unfrozen). The task is a food-safety classification problem — detecting
chemically adulterated mangoes from visually rotten ones.

---

## Running the App

```bash
cd ME1
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**

**Streamlit Cloud URL:** **

---

## Dataset

A brief description of the dataset source is provided in [`dataset/README.md`](dataset/README.md).

**Dataset:** Fruits Disease Dataset — Kaggle  
**Classes:** `Formalin_Mixed`, `Rotten`  
**Source split:** Pre-made `train` / `valid` / `test` splits, Mango sub-folder only

---

## Environment Setup

### Requirements

Python **3.12** is recommended. All dependencies are pinned in [`requirements.txt`](requirements.txt).

### Install locally

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

### Key packages

| Package | Version | Purpose |
|---|---|---|
| `tensorflow` | 2.19.0 | Model training |
| `keras` | 3.15.0 | High-level API |
| `scikit-learn` | 1.8.0 | Evaluation metrics |
| `matplotlib` | 3.10.0 | Plotting |
| `seaborn` | 0.13.2 | Confusion matrix heatmap |
| `streamlit` | 1.60.0 | Web UI |

> **Note:** For faster training, use Google Colab with a T4 GPU runtime
> (`Runtime → Change runtime type → T4 GPU`).

---

## Project Structure

```
ME1/
├── dataset/
│   └── README.md          # Dataset source description
├── model/                 # Saved .keras model files
├── notebooks/
│   └── ME1.ipynb          # Full training pipeline
├── results/               # Plots, confusion matrices, learning curves
├── requirements.txt
├── CONTRIBUTORS.md
└── README.md              # This file
```

---

## Challenges & Solutions

| Challenge | Solution / Notes |
|---|---|
| Dataset contains 5 fruit types and 3 condition classes | Used 3-level parent-folder matching to isolate only `Mango/Rotten` and `Mango/Formalin-mixed` |
| `Formalin-mixed` folder name contains a hyphen | Renamed to `Formalin_Mixed` in the split tree to avoid path issues |
| `valid` folder name differs from convention | Remapped to `val` during split construction |
| Visually similar classes — both show surface deterioration | Fine-tuning top 30 MobileNetV2 layers improves texture-level discrimination |

---

## Possible Improvements

- Add a **Streamlit UI** for live image upload and prediction *(in progress)*
- Experiment with **EfficientNetB0** as an alternative base model
- Extend to all fruit types in the dataset for multi-class food adulteration detection
- Export the model to **TensorFlow Lite** for mobile/edge deployment

---

## Results

| Phase | Val Accuracy | Test Accuracy |
|---|---|---|
| Feature Extraction | — | — |
| Fine-Tuned | — | — |

> Full learning curves and confusion matrices are saved in `results/`.

---

## Contributors

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for the full list of names, GitHub usernames, and registration numbers.

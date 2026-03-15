# 🩺 Comparative Analysis of Machine Learning Models for Early Diabetes Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A professional machine learning research pipeline for early diabetes prediction using the **Pima Indians Diabetes Dataset**, featuring advanced feature engineering, Bayesian hyperparameter optimisation, and a comparative evaluation of seven ML models.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Pipeline Architecture](#-pipeline-architecture)
- [Feature Engineering](#-feature-engineering)
- [Installation](#-installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Research Paper](#-research-paper)
- [Author](#-author)

---

## 🔍 Overview

This project presents a rigorous end-to-end machine learning pipeline for **early diabetes detection** using clinical patient data. The pipeline is modular, reproducible, and designed to maximise predictive accuracy through:

- **Group-wise median imputation** for physiologically implausible zero values
- **8 clinically motivated engineered features** (Metabolic Risk Score, Insulin Sensitivity, etc.)
- **SMOTETomek resampling** to handle class imbalance with boundary cleaning
- **Bayesian hyperparameter optimisation** via Optuna (50 trials per model)
- **Decision threshold optimisation** on a held-out validation set
- **Comparative evaluation** of 7 supervised ML algorithms

---

## 📊 Results

| Model | Accuracy | ROC-AUC | Threshold |
|---|---|---|---|
| **Random Forest 🏆** | **88.31%** | **0.9474** | 0.580 |
| XGBoost | 87.66% | 0.9480 | 0.606 |
| Voting Ensemble | 87.66% | 0.9444 | 0.500 |
| Gradient Boosting | 87.01% | 0.9524 | 0.650 |
| LightGBM | 86.36% | 0.9444 | 0.965 |
| CatBoost | 85.71% | 0.9381 | 0.907 |
| Logistic Regression | 72.73% | 0.8296 | 0.338 |

> **Best Model: Random Forest** — 88.31% accuracy, ROC-AUC 0.9474

### Top Feature Importances (Random Forest)

| Feature | Importance |
|---|---|
| Insulin | 0.3085 |
| Metabolic Risk Score (engineered) | 0.2145 |
| Skin Thickness | 0.1131 |
| Glucose | 0.0842 |
| Glucose-Insulin Ratio (engineered) | 0.0748 |

---

## 📁 Project Structure

```
diabetes-ml-research/
│
├── src/                          # Core pipeline scripts
│   ├── config.py                 # Central configuration (paths, params, model list)
│   ├── step1_prepare_data.py     # Load, impute, engineer features, SMOTE → saved .pkl
│   ├── step2_tune_models.py      # Optuna Bayesian tuning → saved best params .pkl
│   ├── step3_train_evaluate.py   # Train, evaluate, compare all models
│   └── main.py                   # Orchestrates all steps end-to-end
│
├── data/
│   ├── README.md                 # Instructions to download the dataset
│   ├── Diabetes_ML_Paper_IEEE_TwoColumn.docx       
│   ├── Diabetes_ML_Research_Paper.docx                
│   └── diabetes_cleaned.csv      # Preprocessed dataset (add manually — see data/README.md)
│
├── paper/
│   └── README.md                 # Research paper details and download link
│
├── notebooks/
│   └── README.md                 # Notebook usage notes
│
├── requirements.txt              # All Python dependencies
├── .gitignore                    # Files excluded from version control
└── README.md                     # This file
```

---

## 🏗️ Pipeline Architecture

```
diabetes_cleaned.csv
        │
        ▼
┌─────────────────────┐
│  Step 1: Prepare    │  Zero → NaN → Group-wise imputation
│  Data               │  + Feature Engineering (8 new features)
│  [run once]         │  + SMOTETomek resampling
└────────┬────────────┘
         │ prepared_data.pkl
         ▼
┌─────────────────────┐
│  Step 2: Tune       │  Optuna TPE (50 trials × 15-fold CV)
│  Models             │  Scores by ROC-AUC
│  [run once]         │
└────────┬────────────┘
         │ best_params.pkl
         ▼
┌─────────────────────┐
│  Step 3: Train &    │  7 models trained + evaluated
│  Evaluate           │  Threshold optimisation on validation set
│  [run freely]       │  Confusion matrices, ROC-AUC, comparison table
└─────────────────────┘
```

**Key design principle:** Steps 1 and 2 save their outputs as `.pkl` files. `main.py` auto-skips them on subsequent runs, so repeated experiments only re-run Step 3 (fast, ~1–2 minutes).

---

## ⚗️ Feature Engineering

| Feature | Formula | Clinical Rationale |
|---|---|---|
| Glucose-Insulin Ratio | Glucose / (Insulin + 1) | Proxy for insulin resistance |
| BMI × Age | BMI × Age | Obesity risk compounding with age |
| Metabolic Risk Score | Normalised(Glucose + BMI + Insulin) | Composite metabolic health index |
| Insulin Sensitivity | Glucose / (Insulin + ε) | Quantifies cellular insulin response |
| Age × Glucose | Age × Glucose | Age-amplified glucose risk |
| Glucose² | Glucose² | Non-linear risk escalation |
| BMI Category | WHO bins (0–3) | Clinically meaningful obesity banding |
| High Risk Flag | Glucose > 140 AND BMI > 30 | Binary clinical red-flag indicator |

Three low-importance features were dropped after initial analysis: `BloodPressure`, `Pregnancies`, `DiabetesPedigreeFunction`.

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/diabetes-ml-research.git
cd diabetes-ml-research
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Full pipeline (first run)
```bash
cd src
python main.py
```

### Experiment with models only (after first run)
```bash
cd src
python step3_train_evaluate.py
```

### Configuration
All settings are in `src/config.py`:
- `OPTUNA_TRIALS` — increase for better tuning (slower)
- `ACTIVE_MODELS` — comment out models to skip them
- `DROP_FEATURES` — features to exclude from training

---

## 📂 Dataset

The dataset used is the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository / Kaggle.

See [`data/README.md`](data/README.md) for download instructions.

Place the downloaded file as:
```
data/diabetes_cleaned.csv
```

> The dataset is not included in this repository due to redistribution considerations. It is freely available from the sources listed in `data/README.md`.

---

## 📄 Research Paper

A full research paper documenting this study is available in the `paper/` directory.

**Title:** *Comparative Analysis of Machine Learning Models for Early Detection of Diabetes Using the Pima Indians Dataset*

**Sections:**
- Introduction & Literature Review
- Dataset Description
- Methodology (preprocessing, feature engineering, optimisation)
- Experimental Results (accuracy, ROC-AUC, confusion matrices, feature importances)
- Conclusion & Future Work
- References (IEEE format)

See [`paper/README.md`](paper/README.md) for details.

---

## 👤 Author

**Saurabh Sharma**

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Pima Indians Diabetes Dataset — National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/), [Optuna](https://optuna.org/), [imbalanced-learn](https://imbalanced-learn.org/)

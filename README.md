# Telco Customer Churn Prediction (Project 7 – Fall 2025)

Team project for predicting customer churn using the **Telco Customer Churn** dataset (binary classification).

**Primary evaluation focus:** maximize **Recall** for the churn class (`churn = 1`), while keeping **F1 ≥ 0.50** (challenge constraint).

Implemented phases:
- Phase 1: Exploratory Data Analysis (EDA)
- Phase 2: Preprocessing
- Phase 3: Feature Engineering & Feature Selection
- Phase 4: Modeling + Optimization (imbalance handling, CV, tuning, soft voting)

---

## Project structure

```text
.
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   ├── features.py
│   └── modeling.py
├── reports/
│   ├── eda.md
│   ├── preprocessing.md
│   ├── feature_engineering.md
│   ├── modeling.md
│   ├── figures/
│   └── tables/
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── requirements.txt
```

Notes:
- Run all commands from the repo root.
- Do **not** include `.venv/`, `.git/`, `__pycache__/` in the final submission zip.

---

## Setup

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r requirements.txt
```

If `xgboost` installation fails on your machine, the Phase 4 script will still run but will skip the XGBoost model.

---

## Dataset

Place the dataset at:

```text
data/raw/telco_churn.csv
```

Example:

### Linux / macOS
```bash
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv
```

### Windows (PowerShell)
```powershell
Copy-Item "$env:USERPROFILE\Downloads\WA_Fn-UseC_-Telco-Customer-Churn.csv" "data\raw\telco_churn.csv"
```

---

## Phase 1: EDA

Run:
```bash
python -m src.eda
```

Outputs:
- `reports/eda.md`
- `reports/figures/` (plots)

---

## Phase 2: Preprocessing

Run:
```bash
python -m src.preprocess
```

What it does (high level):
- Cleans and encodes the raw data (binary mappings + one-hot encoding)
- Scales numerical features (as configured)
- Produces leakage-safe train/test processed datasets
- Saves preprocessing artifacts for reuse

Outputs (see `reports/preprocessing.md` for details):
- `reports/preprocessing.md`
- `data/processed/` (processed datasets)
- `models/preprocessor.joblib`
- `models/binary_mappings.json`

---

## Phase 3: Feature Engineering & Feature Selection

Run:
```bash
python -m src.features
```

What it does:
- Reuses Phase 2 outputs (leakage-safe)
- Adds engineered features (e.g., charge rate, service count, tenure grouping)
- Runs feature selection with multiple methods (filter + model-based)
- Produces a final selected feature set and documents the rationale

Outputs:
- `reports/feature_engineering.md`
- `reports/tables/selected_features_filter.csv`
- `reports/tables/selected_features_model_l1.csv`
- `reports/tables/selected_features_model_rf.csv`
- `reports/tables/selected_features_final.csv`

---

## Phase 4: Modeling + Optimization

Run:
```bash
python -m src.modeling
```

What it does:
- Handles class imbalance in two ways and compares them:
  - `class_weight="balanced"`
  - **SMOTE** (applied **only on training folds** inside CV)
- Trains and evaluates baseline models with **Stratified K-Fold CV**:
  - Logistic Regression
  - SVM / KNN (depending on configuration)
  - Random Forest
  - Gradient Boosting (XGBoost if available)
- Hyperparameter tuning for at least two models (e.g., LogReg & RF)
- Builds a **Soft Voting** ensemble from the top models
- Produces CV metrics (mean ± std), emphasizing **Recall for churn=1** and tracking F1 for the challenge constraint (F1 ≥ 0.50)

Outputs:
- `reports/modeling.md`
- `reports/tables/phase4_cv_baselines.csv`
- `reports/tables/phase4_cv_tuned.csv`
- `reports/tables/phase4_cv_all_results.csv`
- `models/phase4_best_model.joblib`

---

## Quick verification

```bash
# reports
ls -lah reports/eda.md reports/preprocessing.md reports/feature_engineering.md reports/modeling.md

# key tables / artifacts
ls -lah reports/tables | head
ls -lah data/processed | head
ls -lah models | head
```

---

## Next phase (Phase 5)

Phase 5 (final evaluation) should include:
- Test-set evaluation (Accuracy/Precision/Recall/F1)
- ROC-AUC for top models
- Confusion matrix
- Final model selection + business recommendations

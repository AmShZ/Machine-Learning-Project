# Telco Customer Churn Prediction

Team project for predicting customer churn using the **Telco Customer Churn** dataset (binary classification).

**Primary evaluation focus:** maximize **Recall** for the churn class (`churn = 1`).

Included so far:
- Phase 1: Exploratory Data Analysis (EDA)
- Phase 2: Preprocessing (leakage-safe)
- Phase 3: Feature Engineering & Feature Selection

---

## Project structure

.
├── src/
│   ├── eda.py
│   ├── preprocess.py
│   └── features.py
├── reports/
│   ├── eda.md
│   ├── preprocessing.md
│   ├── feature_engineering.md
│   ├── figures/
│   └── tables/
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── requirements.txt

Notes:
- All scripts are run from the repo root.

---

## Setup (Linux / macOS)

From the repo root:

mkdir -p src reports/figures reports/tables data/raw data/processed models

Create a virtual environment and install dependencies:

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt

---

## Dataset

Put the dataset here:

cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv

---

## Phase 1: EDA

Run:

python3 -m src.eda

Outputs:
- reports/eda.md
- reports/figures/ (plots)

---

## Phase 2: Preprocessing (leakage-safe)

Run:

python3 -m src.preprocess

What it does (high level):
- Converts target to `churn` (Yes=1, No=0)
- Handles missing values (including `TotalCharges`)
- Encodes categorical features (binary mapping + one-hot)
- Scales numeric features
- Uses a leakage-safe protocol: stratified train/test split first (80/20, seed=42), fit preprocessing on train only

Outputs (key artifacts):
- data/processed/telco_churn_clean.csv
- data/processed/ (preprocessed train/test CSVs + train/test indices)
- data/processed/feature_names.txt
- models/preprocessor.joblib
- models/binary_mappings.json
- reports/preprocessing.md

---

## Phase 3: Feature Engineering & Selection

Run:

python3 -m src.features

What it does:
- Reuses Phase 2 split + clean data (no leakage)
- Adds engineered features:
  - AvgChargesPerMonth
  - NumServicesYes
  - TenureGroup
- Runs feature selection:
  - ANOVA F-test (top ~15)
  - L1 Logistic (top ~15 by |coef|)
  - RandomForest importance (top ~15)
- Produces a final selected feature set and documents the logic

Outputs:
- reports/feature_engineering.md
- reports/tables/selected_features_filter.csv
- reports/tables/selected_features_model_l1.csv
- reports/tables/selected_features_model_rf.csv
- reports/tables/selected_features_final.csv

---

## Quick verification

Check reports exist:

ls -lah reports/eda.md reports/preprocessing.md reports/feature_engineering.md

Check phase folders:

ls -lah reports/figures | head
ls -lah reports/tables | head
ls -lah models | head
ls -lah data/processed | head

Optional sanity check for churn labels (adjust the filename to whichever processed CSV you want to inspect):

python3 - << 'PY'
import pandas as pd
path = "data/processed/telco_churn_clean.csv"
df = pd.read_csv(path)
print("loaded:", path)
print("shape:", df.shape)
print("has churn:", "churn" in df.columns)
if "churn" in df.columns:
    print("churn unique:", sorted(df["churn"].dropna().unique().tolist()))
PY

---

## Next (Phase 4+)

Planned next steps:
- Modeling + optimization (with imbalance handling, e.g., SMOTE / class weights)
- Hyperparameter tuning and model selection driven by Recall for churn=1
- Final evaluation report + saved best model under models/
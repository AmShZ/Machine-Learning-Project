# Telco Customer Churn Prediction

Team project for predicting customer churn using the **Telco Customer Churn** dataset (binary classification).

**Primary evaluation focus:** maximize **Recall** for the churn class (`churn = 1`).

Included so far:
- Phase 1: Exploratory Data Analysis (EDA)
- Phase 2: Preprocessing
- Phase 3: Feature Engineering & Feature Selection

---

## Project structure

```text
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
```

Notes:
- All scripts are run from the repo root.

---

## Setup (Linux / macOS)

From the repo root:

```bash
mkdir -p src reports/figures reports/tables data/raw data/processed models
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

---

## Dataset

Put the dataset here:

```bash
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv
```

---

## Phase 1: EDA

Run:

```bash
python3 -m src.eda
```

Outputs:
- `reports/eda.md`
- `reports/figures/` (plots)

---

## Phase 2: Preprocessing

Run:

```bash
python3 -m src.preprocess
```

What it does (high level):
- Cleans/encodes the raw data
- Produces preprocessed datasets and saves preprocessing artifacts for reuse

Outputs (see `reports/preprocessing.md` for the exact list):
- `reports/preprocessing.md`
- `data/processed/` (processed datasets + metadata like feature names / indices if applicable)
- `models/` (saved preprocessor and mappings, if applicable)

---

## Phase 3: Feature Engineering & Feature Selection

Run:

```bash
python3 -m src.features
```

What it does:
- Reuses the Phase 2 outputs (leakage-safe)
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

## Quick verification

Check reports exist:

```bash
ls -lah reports/eda.md reports/preprocessing.md reports/feature_engineering.md
```

Check artifacts exist:

```bash
ls -lah reports/figures | head
ls -lah reports/tables | head
ls -lah data/processed | head
ls -lah models | head
```

Optional: sanity check that a processed CSV contains `churn` and it’s binary  
(adjust the path to the processed file you want to inspect):

```bash
python3 - << 'PY'
import pandas as pd

path = "data/processed/telco_churn_preprocessed.csv"  # change if your output file has a different name
df = pd.read_csv(path)

print("loaded:", path)
print("shape:", df.shape)
print("has churn:", "churn" in df.columns)
if "churn" in df.columns:
    print("churn unique:", sorted(df["churn"].dropna().unique().tolist()))
PY
```

---

## Next phases

Suggested next steps:
- Modeling + optimization (handle imbalance, tune models, focus on Recall for churn=1)
- Final evaluation report and saved best model under `models/`
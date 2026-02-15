# Telco Customer Churn — Phase 1 (EDA) + Phase 2 (Preprocessing)

This repo contains **Phase 1: EDA** and **Phase 2: Preprocessing** for the Telco Customer Churn dataset.

Outputs are lightweight and reproducible:
- EDA generates `reports/eda.md` + plots in `reports/figures/`
- Preprocessing generates a clean ML-ready CSV and saves the fitted preprocessing pipeline

---

## Project structure

```
.
├── src/
│   ├── eda.py
│   └── preprocess.py
├── reports/
│   ├── eda.md
│   ├── preprocessing.md
│   └── figures/
├── data/
│   ├── raw/
│   └── processed/
├── models/
└── requirements.txt
```

`data/` and `models/` are gitignored (recommended).

---

## Setup (Linux)

From the repo root:

```bash
mkdir -p src reports data/raw data/processed models
```

Create venv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

---

## Put the dataset

Recommended (simplest): rename to `telco_churn.csv`:

```bash
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv
```

If you don’t want to rename: keep the original name and pass `--data` when running scripts.

---

## Phase 1 — EDA

Run:

```bash
python -m src.eda
```

Custom dataset path:

```bash
python -m src.eda --data /path/to/your.csv
```

Outputs:
- `reports/eda.md`
- `reports/figures/` (plots)

---

## Phase 2 — Preprocessing

Run:

```bash
python -m src.preprocess
```

Custom dataset path:

```bash
python -m src.preprocess --data /path/to/your.csv
```

Outputs:
- `data/processed/telco_churn_preprocessed.csv` (final features + `churn` target column)
- `data/processed/feature_names.txt`
- `reports/preprocessing.md`
- `models/preprocessor.joblib`
- `models/binary_mappings.json`

---

## What to verify (to make sure the pipeline ran correctly)

### 1) Files exist
After running both phases:

```bash
ls -lah reports/eda.md reports/preprocessing.md
ls -lah reports/figures | head
ls -lah data/processed/telco_churn_preprocessed.csv
ls -lah models/preprocessor.joblib models/binary_mappings.json
```

You should see:
- `reports/eda.md`
- `reports/preprocessing.md`
- multiple `.png` files in `reports/figures/`
- `data/processed/telco_churn_preprocessed.csv`
- `models/preprocessor.joblib`
- `models/binary_mappings.json`

### 2) EDA content looks sane
Open `reports/eda.md` and check:
- dataset row/col counts are printed
- churn rate is printed
- figures are referenced (images show up if you view markdown)

### 3) Preprocessed dataset shape and target
Run:

```bash
python - << 'PY'
import pandas as pd
df = pd.read_csv("data/processed/telco_churn_preprocessed.csv")
print("rows, cols:", df.shape)
print("has churn:", "churn" in df.columns)
print("churn value counts:")
print(df["churn"].value_counts())
PY
```

Expected:
- `has churn: True`
- `churn` contains only `0` and `1`
- row count matches the original dataset (usually 7043)

### 4) No missing values in the final output
Run:

```bash
python - << 'PY'
import pandas as pd
df = pd.read_csv("data/processed/telco_churn_preprocessed.csv")
print("total missing:", int(df.isna().sum().sum()))
PY
```

Expected:
- `total missing: 0`

### 5) Feature names match column count (sanity check)
Run:

```bash
python - << 'PY'
import pandas as pd
fn = open("data/processed/feature_names.txt").read().splitlines()
df = pd.read_csv("data/processed/telco_churn_preprocessed.csv")
print("features in txt:", len(fn))
print("features in csv (excluding churn):", df.shape[1]-1)
PY
```

Expected:
- the two numbers match

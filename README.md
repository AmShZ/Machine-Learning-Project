# Telco Customer Churn Prediction

This repo is a team project around predicting customer churn using the Telco Customer Churn dataset.

Right now it includes:
- Phase 1: Exploratory Data Analysis (EDA)
- Phase 2: Preprocessing


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

Notes:
- `data/` and `models/` are gitignored by default. Keep large files out of git.
- All scripts are run from the repo root.

---

## Setup (Linux)

From the repo root:

```bash
mkdir -p src reports data/raw data/processed models
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

Recommended path and name:

```bash
cp /path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/telco_churn.csv
```

If you do not want to rename the file, keep the original name and pass `--data` when running scripts.

---

## Phase 1: EDA

Run:

```bash
python3 -m src.eda
```

With a custom dataset path:

```bash
python3 -m src.eda --data /path/to/your.csv
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

With a custom dataset path:

```bash
python3 -m src.preprocess --data /path/to/your.csv
```

Outputs:
- `data/processed/telco_churn_preprocessed.csv` (features plus `churn` target)
- `data/processed/feature_names.txt`
- `reports/preprocessing.md`
- `models/preprocessor.joblib`
- `models/binary_mappings.json`

---

## How to verify everything worked

### Check outputs exist

```bash
ls -lah reports/eda.md reports/preprocessing.md
ls -lah reports/figures | head
ls -lah data/processed/telco_churn_preprocessed.csv data/processed/feature_names.txt
ls -lah models/preprocessor.joblib models/binary_mappings.json
```

You should see:
- both report files
- multiple `.png` files under `reports/figures/`
- the processed CSV and feature list
- the saved preprocessing artifacts in `models/`

### Sanity checks on the processed dataset

```bash
python3 - << 'PY'
import pandas as pd

df = pd.read_csv("data/processed/telco_churn_preprocessed.csv")
print("processed shape:", df.shape)
print("has churn:", "churn" in df.columns)
print("churn unique:", sorted(df["churn"].unique().tolist()))
print("total missing:", int(df.isna().sum().sum()))
PY
```

Expected:
- `has churn: True`
- `churn unique: [0, 1]`
- `total missing: 0`
- the number of rows should match the raw dataset (typically 7043 for the common Telco CSV)

### Feature list matches the processed columns

```bash
python3 - << 'PY'
import pandas as pd

fn = open("data/processed/feature_names.txt").read().splitlines()
df = pd.read_csv("data/processed/telco_churn_preprocessed.csv")
print("features in txt:", len(fn))
print("features in csv (excluding churn):", df.shape[1] - 1)
PY
```

Expected:
- the two counts match

---

## Guidance for later phases (for teammates)

The output of Phase 2 is the contract for the rest of the project:
- Use `data/processed/telco_churn_preprocessed.csv` as the default modeling input.
- Treat `churn` as the target.
- If you need the exact preprocessing logic, load `models/preprocessor.joblib`.

Suggested next steps:
- Phase 3: Train baseline models (logistic regression, random forest, gradient boosting)
- Phase 4: Evaluation (train/val split, cross-validation, ROC-AUC, PR-AUC, confusion matrix)
- Phase 5: Model selection and export (save best model under `models/`)
- Phase 6: Inference script (take raw CSV, apply `preprocessor.joblib`, output predictions)

Recommended conventions:
- Put new training code in `src/train.py` and evaluation code in `src/evaluate.py`.
- Save any trained models to `models/`.
- Add new reports to `reports/` and keep them short.
- Do not commit the dataset or large artifacts into git.

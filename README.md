# Telco Customer Churn Prediction

Script-based ML pipeline for predicting customer churn on the Telco dataset.

Primary metric: recall for churn class `1`.
Also reported: precision, F1, and accuracy.

## Project structure

```text
.
|-- src/
|   |-- eda.py
|   |-- preprocess.py
|   |-- features.py
|   |-- modeling.py
|   `-- evaluate.py
|-- reports/
|   |-- eda.md
|   |-- preprocessing.md
|   |-- feature_engineering.md
|   |-- modeling.md
|   |-- final_report.md
|   |-- figures/
|   `-- tables/
|-- data/
|   |-- raw/
|   `-- processed/
|-- models/
`-- requirements.txt
```

## Environment setup

### Linux or macOS

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

## Dataset placement

Place the raw CSV in `data/raw/`.
The scripts first look for `data/raw/telco_churn.csv` and then fall back to `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

## Run order (Phase 1 to Phase 5)

Run all commands from repo root:

```bash
python -m src.eda
python -m src.preprocess
python -m src.features
python -m src.modeling
python -m src.evaluate
```

## Phase outputs

### Phase 1

- `reports/eda.md`
- Figures under `reports/figures/`

### Phase 2

- `data/processed/telco_churn_clean.csv`
- `data/processed/telco_churn_train_preprocessed.csv`
- `data/processed/telco_churn_test_preprocessed.csv`
- `data/processed/telco_churn_preprocessed.csv`
- `data/processed/feature_names.txt`
- `data/processed/train_indices.npy`
- `data/processed/test_indices.npy`
- `models/preprocessor.joblib`
- `models/binary_mappings.json`
- `reports/preprocessing.md`

### Phase 3

- `reports/feature_engineering.md`
- `reports/tables/selected_features_filter.csv`
- `reports/tables/selected_features_model_l1.csv`
- `reports/tables/selected_features_model_rf.csv`
- `reports/tables/selected_features_final.csv`

### Phase 4

- `reports/modeling.md`
- `reports/tables/phase4_cv_baselines.csv`
- `reports/tables/phase4_cv_tuned.csv`
- `reports/tables/phase4_cv_all_results.csv`
- `models/phase4_best_model.joblib`

### Phase 5

- `reports/final_report.md`
- `reports/tables/phase5_test_metrics.csv`
- `reports/tables/phase5_confusion_matrix.csv`
- `reports/tables/phase5_classification_report.csv`
- `reports/tables/phase5_threshold_sweep.csv`
- Optional figures under `reports/figures/` for confusion matrix and threshold tradeoff

## Notes on optional dependencies

- If `imbalanced-learn` is unavailable, SMOTE rows are skipped and logged in Phase 4 outputs.
- If `xgboost` is unavailable, the XGBoost baseline is skipped and logged in Phase 4 outputs.

## Quick verification

```bash
python -m src.eda
python -m src.preprocess
python -m src.features
python -m src.modeling
python -m src.evaluate
```

Then verify key files exist in:

- `reports/`
- `reports/tables/`
- `reports/figures/`
- `data/processed/`
- `models/`

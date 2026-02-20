# Phase 2 - Preprocessing

## Missing values and imputation
- `TotalCharges` blank strings become NaN after numeric conversion.
- For `tenure==0`, `TotalCharges` is set to 0.
- Remaining numeric missing values are median-imputed.

## Encoding and scaling
- Binary categorical columns are label-encoded to 0/1.
- Multi-class categorical columns are one-hot encoded.
- Numeric columns are standardized with `StandardScaler`.

## Leakage-safe protocol
- The train/test split is created first, preprocessing is fit on train only, and saved indices are reused in later phases.

## Summary
| item | value |
| --- | --- |
| TotalCharges NaN after conversion | 11 |
| NaN with tenure==0 | 11 |
| Split | stratified 80/20 (seed=42) |
| Binary label-encoded cols | 5 |
| One-hot encoded cols | 10 |
| Scaled numeric cols | 9 |
| Final feature count | 40 |

## Outputs
- Clean dataset: `data/processed/telco_churn_clean.csv`
- Preprocessed train: `data/processed/telco_churn_train_preprocessed.csv`
- Preprocessed test: `data/processed/telco_churn_test_preprocessed.csv`
- Full preprocessed dataset: `data/processed/telco_churn_preprocessed.csv`
- Feature names: `data/processed/feature_names.txt`
- Train/test indices: `data/processed/train_indices.npy`, `data/processed/test_indices.npy`
- Preprocessor: `models/preprocessor.joblib`
- Binary mappings: `models/binary_mappings.json`
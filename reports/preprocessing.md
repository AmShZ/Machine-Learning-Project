# Phase 2 — Preprocessing

## Missing values & imputation (why)
- `TotalCharges` contains blanks in the raw CSV → becomes NaN after numeric conversion.
- For `tenure==0`, `TotalCharges` should be 0 (new customers have no accumulated charges).
- Remaining NaN are filled with **median** (robust to skew/outliers).

## Encoding & scaling (why)
- Binary categoricals → 0/1 (compact, keeps meaning).
- Multi-class categoricals → one-hot (avoids fake ordinality).
- Numeric features → StandardScaler (helps LR optimization).

## Leakage-safe protocol
- Split first, then fit preprocessing only on the training set. Save split indices for later phases.

## Summary
| item | value |
| --- | --- |
| TotalCharges NaN after conversion | 11 |
| NaN with tenure==0 | 11 |
| Split | stratified 80/20 (seed=42) |
| Binary label-encoded cols | 5 |
| One-hot cols | 10 |
| Scaled numeric cols | 9 |
| Final feature count | 40 |

## Outputs
- Clean (for Phase 3): `data\processed\telco_churn_clean.csv`
- Preprocessed train: `data\processed\telco_churn_train_preprocessed.csv`
- Preprocessed test: `data\processed\telco_churn_test_preprocessed.csv`
- Preprocessor: `models/preprocessor.joblib`
- Mappings: `models/binary_mappings.json`
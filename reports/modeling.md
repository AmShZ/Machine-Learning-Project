# Phase 4 — Modeling & Optimization (Project 7)

## What we did
- Compared **class_weight** vs **SMOTE** for imbalance handling (SMOTE is applied inside CV).
- Trained baseline models: Logistic Regression, SVM (RBF), KNN, Random Forest, and (optionally) XGBoost.
- Hyperparameter tuning for **2 models** (LogReg + RandomForest) using RandomizedSearchCV (optimize Recall).
- Advanced validation: StratifiedKFold (splits=5) and we report mean±std for Recall.

## Key tables
- Baselines: `reports/tables/phase4_cv_baselines.csv`
- Tuned: `reports/tables/phase4_cv_tuned.csv`
- Combined + voting: `reports/tables/phase4_cv_all_results.csv`

## Best model (by Recall, tie-breaker F1)
- **LogReg**

## Notes

## Snapshot (top rows by recall_mean)
| model | imbalance | recall_mean | recall_std | f1_mean | precision_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| LogReg | class_weight | 0.8013 | 0.0379 | 0.6284 | 0.5171 | 0.7487 |
| VotingSoft_top3 | mixed | 0.7946 | 0.0364 | 0.6331 | 0.5265 | 0.7558 |
| LogReg_TUNED | smote | 0.7866 | 0.0393 | 0.6312 | 0.5274 | 0.7563 |
| LogReg | smote | 0.786 | 0.0377 | 0.6315 | 0.5282 | 0.7568 |
| SVM_RBF | class_weight | 0.7719 | 0.0329 | 0.6191 | 0.517 | 0.7481 |
| RandomForest_TUNED | smote | 0.7572 | 0.0259 | 0.6376 | 0.5511 | 0.7716 |
| KNN | smote | 0.7405 | 0.0158 | 0.5584 | 0.4484 | 0.689 |
| SVM_RBF | smote | 0.7211 | 0.0365 | 0.623 | 0.5486 | 0.7687 |
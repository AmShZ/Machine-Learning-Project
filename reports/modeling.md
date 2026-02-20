# Phase 4 - Modeling and Optimization

## Setup
- StratifiedKFold splits: 5
- Primary metric: recall for churn=1
- Feature engineering from Phase 3 applied: yes
- Selected post-onehot feature subset used: yes
- Selected feature count: 15
- Selected feature source: `reports/tables/selected_features_final.csv`

## Imbalance handling
- class_weight=balanced
- SMOTE inside CV pipeline

## Models
- Logistic Regression
- SVM (RBF)
- KNN
- Random Forest
- Gradient boosting: XGBoost when installed

## Hyperparameter tuning
- Tuned models: Logistic Regression and Random Forest
- Search method: RandomizedSearchCV optimized for recall

## Best model
- VotingSoft_top3 (mixed)

## Output tables
- `reports/tables/phase4_cv_baselines.csv`
- `reports/tables/phase4_cv_tuned.csv`
- `reports/tables/phase4_cv_all_results.csv`

## Top CV results
| model | imbalance | recall_mean | recall_std | precision_mean | f1_mean | accuracy_mean |
| --- | --- | --- | --- | --- | --- | --- |
| VotingSoft_top3 | mixed | 0.7953 | 0.0292 | 0.511 | 0.6221 | 0.7437 |
| LogReg_TUNED | smote | 0.7946 | 0.0271 | 0.5057 | 0.6181 | 0.7394 |
| LogReg | smote | 0.7926 | 0.0301 | 0.5132 | 0.6228 | 0.7453 |
| LogReg | class_weight | 0.792 | 0.0274 | 0.5103 | 0.6206 | 0.743 |
| RandomForest_TUNED | smote | 0.7833 | 0.0308 | 0.5336 | 0.6345 | 0.7606 |
| SVM_RBF | class_weight | 0.7739 | 0.0349 | 0.5202 | 0.6218 | 0.7503 |
| SVM_RBF | smote | 0.7612 | 0.0393 | 0.5235 | 0.6198 | 0.7524 |
| KNN | smote | 0.7204 | 0.0089 | 0.497 | 0.5882 | 0.7323 |

## Notes
- All required optional components were available.
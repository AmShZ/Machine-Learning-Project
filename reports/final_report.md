# Final Report - Telco Customer Churn

## Project summary
- Primary objective: maximize recall for churn class (1) with supporting precision, F1, and accuracy.
- Data split policy: fixed stratified split from Phase 2 with seed 42 and reused in Phases 3-5.

## Phase outputs
- Phase 1: `reports/eda.md` and figures in `reports/figures/`
- Phase 2: `reports/preprocessing.md` and processed datasets in `data/processed/`
- Phase 3: `reports/feature_engineering.md` and selected feature tables in `reports/tables/`
- Phase 4: `reports/modeling.md`, CV tables, and `models/phase4_best_model.joblib`
- Phase 5: test evaluation tables and this report

## Phase 4 best CV result
- VotingSoft_top3 (mixed), CV recall=0.7953, CV f1=0.6221

## Phase 5 held-out test metrics
| scenario | threshold | accuracy | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- |
| default_predict | 0.5 | 0.7346 | 0.5 | 0.8075 | 0.6176 |
| selected_threshold | 0.4 | 0.6969 | 0.4627 | 0.8797 | 0.6065 |

## Confusion matrix summary
| scenario | threshold | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- |
| default_predict | 0.5 | 733 | 302 | 72 | 302 |
| selected_threshold | 0.4 | 653 | 382 | 45 | 329 |

## Classification report
| scenario | label | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- | --- |
| default_predict | 0 | 0.9106 | 0.7082 | 0.7967 | 1035.0 |
| default_predict | 1 | 0.5 | 0.8075 | 0.6176 | 374.0 |
| default_predict | accuracy | 0.7346 | 0.7346 | 0.7346 | 0.7346 |
| default_predict | macro avg | 0.7053 | 0.7578 | 0.7072 | 1409.0 |
| default_predict | weighted avg | 0.8016 | 0.7346 | 0.7492 | 1409.0 |
| selected_threshold | 0 | 0.9355 | 0.6309 | 0.7536 | 1035.0 |
| selected_threshold | 1 | 0.4627 | 0.8797 | 0.6065 | 374.0 |
| selected_threshold | accuracy | 0.6969 | 0.6969 | 0.6969 | 0.6969 |
| selected_threshold | macro avg | 0.6991 | 0.7553 | 0.68 | 1409.0 |
| selected_threshold | weighted avg | 0.81 | 0.6969 | 0.7145 | 1409.0 |

## Threshold discussion
- Selected threshold 0.40 by maximizing recall while keeping precision >= 0.450.
- Threshold sweep table: `reports/tables/phase5_threshold_sweep.csv`
- Threshold tradeoff figure: `reports/figures/phase5_threshold_tradeoff.png`

## Key artifacts
- `reports/tables/phase5_test_metrics.csv`
- `reports/tables/phase5_confusion_matrix.csv`
- `reports/tables/phase5_classification_report.csv`
- `reports/final_report.md`

## ROC-AUC comparison (Top 4 models)

Table: `tables/phase5_auc_roc.csv`

![ROC curves](figures/phase5_roc_curves.png)

| model_id | rank | model | imbalance | auc_roc | status |
| --- | --- | --- | --- | --- | --- |
| rank1 | 1 | VotingSoft_top3 | mixed | 0.8387 | ok |
| rank2 | 2 | LogReg_TUNED | smote | 0.8357 | ok |
| rank3 | 3 | LogReg | smote | 0.8391 | ok |
| rank4 | 4 | LogReg | class_weight | 0.8389 | ok |

## Winner model and most important features

Winner model artifact: `models/phase4_best_model.joblib`

Highest-impact features: tenure, Contract_Two year, Contract_Month-to-month, InternetService_Fiber optic, PaymentMethod_Electronic check

Feature table: `tables/phase5_top_features.csv`

| feature | importance |
| --- | --- |
| tenure | 0.163357 |
| Contract_Two year | 0.132942 |
| Contract_Month-to-month | 0.118709 |
| InternetService_Fiber optic | 0.077878 |
| PaymentMethod_Electronic check | 0.070751 |
| OnlineSecurity_No | 0.068744 |
| TenureGroup_48-72 | 0.062283 |
| TechSupport_No | 0.05964 |
| TenureGroup_12-24 | 0.050381 |
| TotalCharges | 0.042654 |
| PaperlessBilling | 0.03836 |
| AvgChargesPerMonth | 0.033153 |
| OnlineBackup_No | 0.028637 |
| TenureGroup_0-12 | 0.026542 |
| MonthlyCharges | 0.025971 |

## Business recommendations
- Prioritize month-to-month customers with renewal incentives and migration paths to longer contracts.
- Bundle security, backup, and support services for high-risk accounts to reduce service-related churn.
- Launch a first-year retention program with proactive outreach during early-tenure periods.
- Target electronic-check customers with autopay conversion campaigns and billing support.
- Use personalized pricing reviews for customers with high monthly or per-month charge intensity.
# Phase 3 - Feature Engineering and Selection

## Engineered features
- `AvgChargesPerMonth = TotalCharges / max(tenure, 1)`
- `NumServicesYes` counts active services with value `Yes`
- `TenureGroup` bins tenure into lifecycle bands

## Selection methods
- Filter method: ANOVA F-test (SelectKBest, k=15)
- Embedded method: L1 Logistic Regression (top 15 by absolute coefficient)
- Model-based method: RandomForest importance (top 15)

## Final subset rule
- The final list uses rank aggregation across all three selectors.

## Final selected features
| feature |
| --- |
| tenure |
| Contract_Month-to-month |
| OnlineSecurity_No |
| InternetService_Fiber optic |
| Contract_Two year |
| TechSupport_No |
| PaymentMethod_Electronic check |
| TotalCharges |
| TenureGroup_0-12 |
| TenureGroup_48-72 |
| PaperlessBilling |
| AvgChargesPerMonth |
| MonthlyCharges |
| TenureGroup_12-24 |
| OnlineBackup_No |

## Output tables
- `reports/tables/selected_features_filter.csv`
- `reports/tables/selected_features_model_l1.csv`
- `reports/tables/selected_features_model_rf.csv`
- `reports/tables/selected_features_final.csv`
# Phase 3 — Feature Engineering & Selection

## Engineered features (what & why)
- **AvgChargesPerMonth** = TotalCharges / max(tenure, 1): captures spending intensity.
- **NumServicesYes**: number of subscribed services; proxy for customer stickiness.
- **TenureGroup**: tenure bins to capture non-linear lifecycle effects.

## Selection methods (top-k)
- ANOVA F-test (SelectKBest), k=15
- L1 Logistic Regression (top-k by |coef|), k=15
- RandomForest (top-k by importance), k=15

## Final subset rationale
We prefer features that are consistently strong across multiple selectors. Final list is ranked by summed ranks across ANOVA + L1 + RF.

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
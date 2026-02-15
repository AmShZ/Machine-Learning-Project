# Phase 3 — Feature Engineering & Selection

## Engineered features
- AvgChargesPerMonth
- NumServicesYes
- TenureGroup

## Leakage-safe protocol
- Train/test split first; preprocessing fit only on train.

## Feature selection
- ANOVA F-test (top 15)
- L1 Logistic |coef| (top 15)
- RandomForest importance (top 15)

## Final rule
- Union, ranked by summed ranks, keep top 15.
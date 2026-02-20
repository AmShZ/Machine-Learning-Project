import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from src.pipeline_utils import (
    add_engineered_features,
    apply_binary_mappings,
    build_preprocessor,
    load_binary_mappings,
)


RANDOM_SEED = 42


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=str, default="data/processed/telco_churn_clean.csv")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    clean_path = Path(args.clean)
    processed_dir = Path(args.processed_dir)
    models_dir = Path(args.models)
    reports_dir = Path(args.reports)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    train_idx_path = processed_dir / "train_indices.npy"
    test_idx_path = processed_dir / "test_indices.npy"
    if not clean_path.exists() or not train_idx_path.exists() or not test_idx_path.exists():
        raise FileNotFoundError("Run Phase 2 first: python -m src.preprocess")

    df = pd.read_csv(clean_path)
    df = add_engineered_features(df)

    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"])

    mappings = load_binary_mappings(models_dir / "binary_mappings.json")
    X = apply_binary_mappings(X, mappings)

    train_idx = np.load(train_idx_path)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]

    preprocessor, _, _ = build_preprocessor(X_train)
    X_train_transformed = preprocessor.fit_transform(X_train)
    feature_names = list(preprocessor.get_feature_names_out())

    k = min(args.k, len(feature_names))

    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(X_train_transformed, y_train)
    filter_idx = np.argsort(skb.scores_)[::-1][:k]
    filter_features = [feature_names[i] for i in filter_idx]

    l1_model = LogisticRegression(
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        max_iter=5000,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    l1_model.fit(X_train_transformed, y_train)
    l1_scores = np.abs(l1_model.coef_).ravel()
    l1_idx = np.argsort(l1_scores)[::-1][:k]
    l1_features = [feature_names[i] for i in l1_idx]

    rf_model = RandomForestClassifier(
        n_estimators=600,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf_model.fit(X_train_transformed, y_train)
    rf_scores = rf_model.feature_importances_
    rf_idx = np.argsort(rf_scores)[::-1][:k]
    rf_features = [feature_names[i] for i in rf_idx]

    def rank_map(features: list[str]) -> dict[str, int]:
        return {feature: rank for rank, feature in enumerate(features, start=1)}

    filter_ranks = rank_map(filter_features)
    l1_ranks = rank_map(l1_features)
    rf_ranks = rank_map(rf_features)

    all_features = sorted(set(filter_features) | set(l1_features) | set(rf_features))
    max_rank = 10_000
    scored = sorted(
        [(feature, filter_ranks.get(feature, max_rank) + l1_ranks.get(feature, max_rank) + rf_ranks.get(feature, max_rank)) for feature in all_features],
        key=lambda item: item[1],
    )
    final_features = [feature for feature, _ in scored[:k]]

    pd.DataFrame({"feature": filter_features}).to_csv(tables_dir / "selected_features_filter.csv", index=False)
    pd.DataFrame({"feature": l1_features}).to_csv(tables_dir / "selected_features_model_l1.csv", index=False)
    pd.DataFrame({"feature": rf_features}).to_csv(tables_dir / "selected_features_model_rf.csv", index=False)
    pd.DataFrame({"feature": final_features}).to_csv(tables_dir / "selected_features_final.csv", index=False)

    lines: list[str] = []
    lines.append("# Phase 3 - Feature Engineering and Selection")
    lines.append("")
    lines.append("## Engineered features")
    lines.append("- `AvgChargesPerMonth = TotalCharges / max(tenure, 1)`")
    lines.append("- `NumServicesYes` counts active services with value `Yes`")
    lines.append("- `TenureGroup` bins tenure into lifecycle bands")
    lines.append("")
    lines.append("## Selection methods")
    lines.append(f"- Filter method: ANOVA F-test (SelectKBest, k={k})")
    lines.append(f"- Embedded method: L1 Logistic Regression (top {k} by absolute coefficient)")
    lines.append(f"- Model-based method: RandomForest importance (top {k})")
    lines.append("")
    lines.append("## Final subset rule")
    lines.append("- The final list uses rank aggregation across all three selectors.")
    lines.append("")
    lines.append("## Final selected features")
    lines.append(markdown_table(pd.DataFrame({"feature": final_features})))
    lines.append("")
    lines.append("## Output tables")
    lines.append("- `reports/tables/selected_features_filter.csv`")
    lines.append("- `reports/tables/selected_features_model_l1.csv`")
    lines.append("- `reports/tables/selected_features_model_rf.csv`")
    lines.append("- `reports/tables/selected_features_final.csv`")

    (reports_dir / "feature_engineering.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

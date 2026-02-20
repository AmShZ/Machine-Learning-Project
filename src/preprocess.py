import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from src.pipeline_utils import (
    apply_binary_mappings,
    build_preprocessor,
    make_binary_mappings,
    resolve_raw_data_path,
)


RANDOM_SEED = 42
TEST_SIZE = 0.20


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def to_feature_frame(transformed, feature_names: list[str]) -> pd.DataFrame:
    if isinstance(transformed, pd.DataFrame):
        return transformed.reset_index(drop=True)
    return pd.DataFrame(transformed, columns=feature_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    data_path = resolve_raw_data_path(args.data)
    out_dir = Path(args.out)
    models_dir = Path(args.models)
    reports_dir = Path(args.reports)

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    total_charges_missing = int(df["TotalCharges"].isna().sum())
    missing_tenure_zero = int((df["TotalCharges"].isna() & (df["tenure"] == 0)).sum())

    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    clean_df = X.copy()
    clean_df["churn"] = y.to_numpy()
    clean_path = out_dir / "telco_churn_clean.csv"
    clean_df.to_csv(clean_path, index=False)

    binary_cols = [column for column in X.columns if X[column].dtype == "object" and X[column].nunique(dropna=True) == 2]
    mappings = make_binary_mappings(X, binary_cols)
    X_mapped = apply_binary_mappings(X, mappings)

    idx = np.arange(len(X_mapped))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y.to_numpy(),
    )

    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "test_indices.npy", test_idx)

    X_train = X_mapped.iloc[train_idx].reset_index(drop=True)
    X_test = X_mapped.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())

    train_features = to_feature_frame(X_train_t, feature_names)
    test_features = to_feature_frame(X_test_t, feature_names)

    train_df = train_features.copy()
    train_df["churn"] = y_train.to_numpy()
    test_df = test_features.copy()
    test_df["churn"] = y_test.to_numpy()

    train_out = out_dir / "telco_churn_train_preprocessed.csv"
    test_out = out_dir / "telco_churn_test_preprocessed.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    full_t = preprocessor.transform(X_mapped)
    full_features = to_feature_frame(full_t, feature_names)
    full_df = full_features.copy()
    full_df["churn"] = y.to_numpy()
    full_df.to_csv(out_dir / "telco_churn_preprocessed.csv", index=False)

    (out_dir / "feature_names.txt").write_text("\n".join(feature_names), encoding="utf-8")
    (models_dir / "binary_mappings.json").write_text(json.dumps(mappings, indent=2), encoding="utf-8")
    dump(preprocessor, models_dir / "preprocessor.joblib")

    overview = pd.DataFrame(
        [
            ["TotalCharges NaN after conversion", total_charges_missing],
            ["NaN with tenure==0", missing_tenure_zero],
            ["Split", f"stratified {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} (seed={args.seed})"],
            ["Binary label-encoded cols", len(binary_cols)],
            ["One-hot encoded cols", len(categorical_cols)],
            ["Scaled numeric cols", len(numeric_cols)],
            ["Final feature count", len(feature_names)],
        ],
        columns=["item", "value"],
    )

    lines: list[str] = []
    lines.append("# Phase 2 - Preprocessing")
    lines.append("")
    lines.append("## Missing values and imputation")
    lines.append("- `TotalCharges` blank strings become NaN after numeric conversion.")
    lines.append("- For `tenure==0`, `TotalCharges` is set to 0.")
    lines.append("- Remaining numeric missing values are median-imputed.")
    lines.append("")
    lines.append("## Encoding and scaling")
    lines.append("- Binary categorical columns are label-encoded to 0/1.")
    lines.append("- Multi-class categorical columns are one-hot encoded.")
    lines.append("- Numeric columns are standardized with `StandardScaler`.")
    lines.append("")
    lines.append("## Leakage-safe protocol")
    lines.append("- The train/test split is created first, preprocessing is fit on train only, and saved indices are reused in later phases.")
    lines.append("")
    lines.append("## Summary")
    lines.append(markdown_table(overview))
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Clean dataset: `{clean_path.as_posix()}`")
    lines.append(f"- Preprocessed train: `{train_out.as_posix()}`")
    lines.append(f"- Preprocessed test: `{test_out.as_posix()}`")
    lines.append("- Full preprocessed dataset: `data/processed/telco_churn_preprocessed.csv`")
    lines.append("- Feature names: `data/processed/feature_names.txt`")
    lines.append("- Train/test indices: `data/processed/train_indices.npy`, `data/processed/test_indices.npy`")
    lines.append("- Preprocessor: `models/preprocessor.joblib`")
    lines.append("- Binary mappings: `models/binary_mappings.json`")

    (reports_dir / "preprocessing.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

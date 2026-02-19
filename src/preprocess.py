import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
TEST_SIZE = 0.20


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def make_binary_mappings(df: pd.DataFrame, cols: list[str]) -> dict:
    mappings = {}
    for c in cols:
        values = list(pd.Series(df[c].dropna().unique()))
        s = set(values)
        if s == {"Yes", "No"}:
            mapping = {"No": 0, "Yes": 1}
        elif s == {"Female", "Male"}:
            mapping = {"Female": 0, "Male": 1}
        else:
            ordered = sorted(values)
            mapping = {ordered[0]: 0, ordered[1]: 1}
        mappings[c] = mapping
    return mappings


def apply_binary_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    out = df.copy()
    for c, mapping in mappings.items():
        if c in out.columns:
            out[c] = out[c].map(mapping).astype("int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test_size", type=float, default=TEST_SIZE)
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    models_dir = Path(args.models)
    reports_dir = Path(args.reports)

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    totalcharges_missing = int(df["TotalCharges"].isna().sum())
    missing_tenure0 = int((df["TotalCharges"].isna() & (df["tenure"] == 0)).sum())

    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().any():
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")
    clean_df = X.copy()
    clean_df["churn"] = y.to_numpy()
    clean_path = out_dir / "telco_churn_clean.csv"
    clean_df.to_csv(clean_path, index=False)
    binary_cols = [c for c in X.columns if X[c].dtype == "object" and X[c].nunique(dropna=True) == 2]
    mappings = make_binary_mappings(X, binary_cols)
    X_mapped = apply_binary_mappings(X, mappings)
    idx = np.arange(len(X_mapped))
    train_idx, test_idx = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y.to_numpy()
    )
    np.save(out_dir / "train_indices.npy", train_idx)
    np.save(out_dir / "test_indices.npy", test_idx)

    X_train, X_test = X_mapped.iloc[train_idx], X_mapped.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    categorical_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    numeric_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = list(preprocessor.get_feature_names_out())
    train_df = pd.DataFrame(X_train_t, columns=feature_names)
    train_df["churn"] = y_train.to_numpy()
    test_df = pd.DataFrame(X_test_t, columns=feature_names)
    test_df["churn"] = y_test.to_numpy()

    train_out = out_dir / "telco_churn_train_preprocessed.csv"
    test_out = out_dir / "telco_churn_test_preprocessed.csv"
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    full_t = preprocessor.transform(X_mapped)
    full_df = pd.DataFrame(full_t, columns=feature_names)
    full_df["churn"] = y.to_numpy()
    full_df.to_csv(out_dir / "telco_churn_preprocessed.csv", index=False)

    (out_dir / "feature_names.txt").write_text("\n".join(feature_names), encoding="utf-8")
    (models_dir / "binary_mappings.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(preprocessor, models_dir / "preprocessor.joblib")

    overview = pd.DataFrame(
        [
            ["TotalCharges NaN after conversion", totalcharges_missing],
            ["NaN with tenure==0", missing_tenure0],
            ["Split", f"stratified {int((1-args.test_size)*100)}/{int(args.test_size*100)} (seed={args.seed})"],
            ["Binary label-encoded cols", len(binary_cols)],
            ["One-hot cols", len(categorical_cols)],
            ["Scaled numeric cols", len(numeric_cols)],
            ["Final feature count", len(feature_names)],
        ],
        columns=["item", "value"],
    )

    lines = []
    lines.append("# Phase 2 — Preprocessing")
    lines.append("")
    lines.append("## Missing values & imputation (why)")
    lines.append(
        "- `TotalCharges` contains blanks in the raw CSV → becomes NaN after numeric conversion.\n"
        "- For `tenure==0`, `TotalCharges` should be 0 (new customers have no accumulated charges).\n"
        "- Remaining NaN are filled with **median** (robust to skew/outliers)."
    )
    lines.append("")
    lines.append("## Encoding & scaling (why)")
    lines.append(
        "- Binary categoricals → 0/1 (compact, keeps meaning).\n"
        "- Multi-class categoricals → one-hot (avoids fake ordinality).\n"
        "- Numeric features → StandardScaler (helps LR optimization)."
    )
    lines.append("")
    lines.append("## Leakage-safe protocol")
    lines.append("- Split first, then fit preprocessing only on the training set. Save split indices for later phases.")
    lines.append("")
    lines.append("## Summary")
    lines.append(markdown_table(overview))
    lines.append("")
    lines.append("## Outputs")
    lines.append(f"- Clean (for Phase 3): `{clean_path}`")
    lines.append(f"- Preprocessed train: `{train_out}`")
    lines.append(f"- Preprocessed test: `{test_out}`")
    lines.append(f"- Preprocessor: `models/preprocessor.joblib`")
    lines.append(f"- Mappings: `models/binary_mappings.json`")

    (reports_dir / "preprocessing.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

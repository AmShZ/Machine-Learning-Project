import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
        out[c] = out[c].map(mapping).astype("int64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--out", type=str, default="data/processed")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--reports", type=str, default="reports")
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
    X = df.drop(columns=["Churn", "customerID"])

    binary_cols = [c for c in X.columns if X[c].dtype == "object" and X[c].nunique(dropna=True) == 2]
    mappings = make_binary_mappings(X, binary_cols)
    X = apply_binary_mappings(X, mappings)

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_t = preprocessor.fit_transform(X)
    feature_names = list(preprocessor.get_feature_names_out())

    X_df = pd.DataFrame(X_t, columns=feature_names, index=df.index)
    out_df = X_df.copy()
    out_df["churn"] = y.to_numpy()

    out_path = out_dir / "telco_churn_preprocessed.csv"
    out_df.to_csv(out_path, index=False)

    (out_dir / "feature_names.txt").write_text("\n".join(feature_names), encoding="utf-8")
    (models_dir / "binary_mappings.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    dump(preprocessor, models_dir / "preprocessor.joblib")

    overview = pd.DataFrame(
        [
            ["rows", df.shape[0]],
            ["raw cols", df.shape[1]],
            ["target", "Churn (Yes/No) → churn (1/0)"],
            ["TotalCharges missing after conversion", totalcharges_missing],
            ["of which tenure==0", missing_tenure0],
            ["binary encoded cols", len(binary_cols)],
            ["one-hot cols", len(categorical_cols)],
            ["numeric scaled cols", len(numeric_cols)],
            ["final feature count", len(feature_names)],
            ["output", str(out_path)],
        ],
        columns=["item", "value"],
    )

    cols_tbl = pd.DataFrame(
        [
            ["binary_label", ", ".join(binary_cols) if binary_cols else "-"],
            ["one_hot", ", ".join(categorical_cols) if categorical_cols else "-"],
            ["numeric_scaled", ", ".join(numeric_cols) if numeric_cols else "-"],
        ],
        columns=["group", "columns"],
    )

    report_lines = []
    report_lines.append("# Phase 2 — Preprocessing")
    report_lines.append("")
    report_lines.append("## What we did")
    report_lines.append("- Converted `TotalCharges` to numeric (blank strings → NaN)")
    report_lines.append("- Filled NaN in `TotalCharges` for `tenure==0` with 0 (any remaining NaN → median)")
    report_lines.append("- Converted `Churn` to `churn` (Yes=1, No=0)")
    report_lines.append("- Label-encoded binary categorical columns (0/1)")
    report_lines.append("- One-hot encoded multi-class categorical columns")
    report_lines.append("- StandardScaled numeric columns")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append(markdown_table(overview))
    report_lines.append("")
    report_lines.append("## Column grouping")
    report_lines.append(markdown_table(cols_tbl))
    report_lines.append("")
    report_lines.append("## Outputs")
    report_lines.append(f"- `{out_path}`")
    report_lines.append(f"- `data/processed/feature_names.txt`")
    report_lines.append(f"- `models/preprocessor.joblib`")
    report_lines.append(f"- `models/binary_mappings.json`")

    (reports_dir / "preprocessing.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

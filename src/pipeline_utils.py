from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_DATA_CANDIDATES = (
    "data/raw/telco_churn.csv",
    "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
)


def resolve_raw_data_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    for candidate in RAW_DATA_CANDIDATES:
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path
    raise FileNotFoundError(
        f"Raw dataset not found at '{path.as_posix()}'. "
        f"Tried: {', '.join(RAW_DATA_CANDIDATES)}"
    )


def make_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_binary_mappings(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, int]]:
    mappings: dict[str, dict[str, int]] = {}
    for column in cols:
        values = list(pd.Series(df[column].dropna().unique()))
        values_set = set(values)
        if values_set == {"Yes", "No"}:
            mapping = {"No": 0, "Yes": 1}
        elif values_set == {"Female", "Male"}:
            mapping = {"Female": 0, "Male": 1}
        else:
            ordered = sorted(values)
            if len(ordered) != 2:
                continue
            mapping = {ordered[0]: 0, ordered[1]: 1}
        mappings[column] = mapping
    return mappings


def apply_binary_mappings(df: pd.DataFrame, mappings: dict[str, dict[str, int]]) -> pd.DataFrame:
    out = df.copy()
    for column, mapping in mappings.items():
        if column not in out.columns:
            continue
        mapped = out[column].map(mapping)
        if mapped.isna().any():
            out[column] = mapped.astype("Int64")
        else:
            out[column] = mapped.astype("int64")
    return out


def load_binary_mappings(path: str | Path) -> dict[str, dict[str, int]]:
    mapping_path = Path(path)
    if not mapping_path.exists():
        return {}
    return json.loads(mapping_path.read_text(encoding="utf-8"))


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        out["TotalCharges"] = out["TotalCharges"].fillna(out["TotalCharges"].median())
    if "tenure" in out.columns and "TotalCharges" in out.columns:
        denominator = out["tenure"].replace(0, 1)
        out["AvgChargesPerMonth"] = out["TotalCharges"] / denominator
    else:
        out["AvgChargesPerMonth"] = 0.0
    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present = [column for column in service_cols if column in out.columns]
    if present:
        out["NumServicesYes"] = (out[present] == "Yes").sum(axis=1)
    else:
        out["NumServicesYes"] = 0
    if "tenure" in out.columns:
        out["TenureGroup"] = pd.cut(
            out["tenure"],
            bins=[-0.1, 12, 24, 48, 72, 10_000],
            labels=["0-12", "12-24", "24-48", "48-72", "72+"],
        ).astype("object")
    else:
        out["TenureGroup"] = "unknown"
    return out


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = [column for column in X_train.columns if X_train[column].dtype == "object"]
    numeric_cols = [column for column in X_train.columns if X_train[column].dtype != "object"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    if hasattr(preprocessor, "set_output"):
        preprocessor.set_output(transform="pandas")
    return preprocessor, numeric_cols, categorical_cols


def load_selected_features(path: str | Path) -> list[str]:
    selected_path = Path(path)
    if not selected_path.exists():
        return []
    table = pd.read_csv(selected_path)
    if "feature" not in table.columns:
        return []
    return table["feature"].dropna().astype(str).tolist()


class FeatureSubsetTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features: list[str] | None = None):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        if self.selected_features is None:
            self.features_ = list(X.columns) if hasattr(X, "columns") else None
            return self
        if not hasattr(X, "columns"):
            raise ValueError("Selected feature filtering requires DataFrame output from preprocessing.")
        present = [feature for feature in self.selected_features if feature in X.columns]
        if not present:
            raise ValueError("None of the selected features were found after preprocessing.")
        self.features_ = present
        return self

    def transform(self, X):
        if self.features_ is None:
            return X
        if not hasattr(X, "loc"):
            raise ValueError("Selected feature filtering requires DataFrame input.")
        return X.loc[:, self.features_]

    def get_feature_names_out(self, input_features=None):
        if getattr(self, "features_", None) is None:
            if input_features is None:
                return np.array([], dtype=object)
            return np.asarray(input_features, dtype=object)
        return np.asarray(self.features_, dtype=object)

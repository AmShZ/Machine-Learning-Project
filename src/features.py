import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


RANDOM_SEED = 42


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom = out["tenure"].replace(0, 1)
    out["AvgChargesPerMonth"] = out["TotalCharges"] / denom
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    present = [c for c in service_cols if c in out.columns]
    out["NumServicesYes"] = (out[present] == "Yes").sum(axis=1) if present else 0
    out["TenureGroup"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 12, 24, 48, 72, 10_000],
        labels=["0-12", "12-24", "24-48", "48-72", "72+"],
    ).astype("object")
    return out


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
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
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["TotalCharges"].median())
    df = add_engineered_features(df)
    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"])
    mappings_path = models_dir / "binary_mappings.json"
    if mappings_path.exists():
        mappings = json.loads(mappings_path.read_text(encoding="utf-8"))
        for c, mapping in mappings.items():
            if c in X.columns:
                X[c] = X[c].map(mapping).astype("int64")

    train_idx = np.load(train_idx_path)
    test_idx = np.load(test_idx_path)
    X_train, y_train = X.iloc[train_idx].copy(), y[train_idx]
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

    Xtr = preprocessor.fit_transform(X_train)
    feature_names = list(preprocessor.get_feature_names_out())
    k = min(args.k, Xtr.shape[1])
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(Xtr, y_train)
    filt_idx = np.argsort(skb.scores_)[::-1][:k]
    filt_features = [feature_names[i] for i in filt_idx]
    lr = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    lr.fit(Xtr, y_train)
    coef = np.abs(lr.coef_).ravel()
    l1_idx = np.argsort(coef)[::-1][:k]
    l1_features = [feature_names[i] for i in l1_idx]
    rf = RandomForestClassifier(
        n_estimators=600, random_state=RANDOM_SEED,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(Xtr, y_train)
    imp = rf.feature_importances_
    rf_idx = np.argsort(imp)[::-1][:k]
    rf_features = [feature_names[i] for i in rf_idx]
    def ranks(lst):
        return {f: r for r, f in enumerate(lst, start=1)}

    r1, r2, r3 = ranks(filt_features), ranks(l1_features), ranks(rf_features)
    all_feats = sorted(set(filt_features) | set(l1_features) | set(rf_features))
    big = 10_000
    scored = sorted([(f, r1.get(f, big) + r2.get(f, big) + r3.get(f, big)) for f in all_feats], key=lambda x: x[1])
    final_features = [f for f, _ in scored[:k]]
    pd.DataFrame({"feature": filt_features}).to_csv(tables_dir / "selected_features_filter.csv", index=False)
    pd.DataFrame({"feature": l1_features}).to_csv(tables_dir / "selected_features_model_l1.csv", index=False)
    pd.DataFrame({"feature": rf_features}).to_csv(tables_dir / "selected_features_model_rf.csv", index=False)
    pd.DataFrame({"feature": final_features}).to_csv(tables_dir / "selected_features_final.csv", index=False)

    lines = []
    lines.append("# Phase 3 — Feature Engineering & Selection")
    lines.append("")
    lines.append("## Engineered features (what & why)")
    lines.append("- **AvgChargesPerMonth** = TotalCharges / max(tenure, 1): captures spending intensity.")
    lines.append("- **NumServicesYes**: number of subscribed services; proxy for customer stickiness.")
    lines.append("- **TenureGroup**: tenure bins to capture non-linear lifecycle effects.")
    lines.append("")
    lines.append("## Selection methods (top-k)")
    lines.append(f"- ANOVA F-test (SelectKBest), k={k}")
    lines.append(f"- L1 Logistic Regression (top-k by |coef|), k={k}")
    lines.append(f"- RandomForest (top-k by importance), k={k}")
    lines.append("")
    lines.append("## Final subset rationale")
    lines.append("We prefer features that are consistently strong across multiple selectors. "
                 "Final list is ranked by summed ranks across ANOVA + L1 + RF.")
    lines.append("")
    lines.append("## Final selected features")
    lines.append(markdown_table(pd.DataFrame({"feature": final_features})))

    (reports_dir / "feature_engineering.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
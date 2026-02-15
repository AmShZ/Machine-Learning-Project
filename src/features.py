import argparse
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
from sklearn.model_selection import train_test_split


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Clean TotalCharges
    out["TotalCharges_num"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out.loc[out["TotalCharges_num"].isna() & (out["tenure"] == 0), "TotalCharges_num"] = 0.0
    out["TotalCharges_num"] = out["TotalCharges_num"].fillna(out["TotalCharges_num"].median())

    # (1) Avg charges per month implied by total/tenure
    denom = out["tenure"].replace(0, 1)
    out["AvgChargesPerMonth"] = out["TotalCharges_num"] / denom

    # (2) Count how many service-related features are "Yes"
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    present = [c for c in service_cols if c in out.columns]
    out["NumServicesYes"] = (out[present] == "Yes").sum(axis=1) if present else 0

    # (3) Tenure group bucket
    out["TenureGroup"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 12, 24, 48, 72, 10_000],
        labels=["0-12", "12-24", "24-48", "48-72", "72+"],
    ).astype("object")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--tables", type=str, default="results/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=15)
    args = parser.parse_args()

    data_path = Path(args.data)
    tables_dir = Path(args.tables)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    df = add_engineered_features(df)

    y = (df["Churn"] == "Yes").astype(int).to_numpy()
    X = df.drop(columns=["Churn", "customerID"], errors="ignore")

    categorical_cols = [c for c in X.columns if X[c].dtype == "object"]
    numeric_cols = [c for c in X.columns if X[c].dtype != "object"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    Xtr = preprocessor.fit_transform(X_train)
    feature_names = list(preprocessor.get_feature_names_out())

    k = min(args.k, Xtr.shape[1])

    # Filter: ANOVA F-test
    skb = SelectKBest(score_func=f_classif, k=k)
    skb.fit(Xtr, y_train)
    filt_idx = np.argsort(skb.scores_)[::-1][:k]
    filt_features = [feature_names[i] for i in filt_idx]

    # Model-based: L1 logistic
    lr = LogisticRegression(
        penalty="l1", solver="saga", max_iter=5000,
        class_weight="balanced", random_state=args.seed, n_jobs=-1
    )
    lr.fit(Xtr, y_train)
    coef = np.abs(lr.coef_).ravel()
    l1_idx = np.argsort(coef)[::-1][:k]
    l1_features = [feature_names[i] for i in l1_idx]

    # Model-based: Random Forest importance
    rf = RandomForestClassifier(
        n_estimators=600, random_state=args.seed,
        class_weight="balanced", n_jobs=-1
    )
    rf.fit(Xtr, y_train)
    imp = rf.feature_importances_
    rf_idx = np.argsort(imp)[::-1][:k]
    rf_features = [feature_names[i] for i in rf_idx]

    # rank by summed ranks across methods
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

    (tables_dir / "feature_engineering_notes.md").write_text(
        "\n".join([
            "# Phase 3 — Feature Engineering & Selection",
            "",
            "## Engineered features",
            "- AvgChargesPerMonth",
            "- NumServicesYes",
            "- TenureGroup",
            "",
            "## Leakage-safe protocol",
            "- Train/test split first; preprocessing fit only on train.",
            "",
            "## Feature selection",
            f"- ANOVA F-test (top {k})",
            f"- L1 Logistic |coef| (top {k})",
            f"- RandomForest importance (top {k})",
            "",
            "## Final rule",
            f"- Union, ranked by summed ranks, keep top {k}."
        ]),
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
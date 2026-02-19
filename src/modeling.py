import argparse
from pathlib import Path
import json

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Optional: XGBoost (recommended by the project statement)
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE  # type: ignore
    from imblearn.pipeline import Pipeline as ImbPipeline  # type: ignore
    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


RANDOM_SEED = 42


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def apply_binary_mappings(df: pd.DataFrame, mappings: dict) -> pd.DataFrame:
    out = df.copy()
    for col, mp in mappings.items():
        if col in out.columns:
            out[col] = out[col].map(mp).astype("Int64")
    return out


def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
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
    return preprocessor


def cv_eval(pipeline, X, y, cv, scorers: dict) -> dict:
    res = cross_validate(
        pipeline,
        X, y,
        cv=cv,
        scoring=scorers,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    out = {}
    for k, v in res.items():
        if k.startswith("test_"):
            metric = k.replace("test_", "")
            out[f"{metric}_mean"] = float(np.mean(v))
            out[f"{metric}_std"] = float(np.std(v))
    return out


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Modeling + Optimization (Project 7 - Telco Churn)")
    parser.add_argument("--data", type=str, default="data/processed/telco_churn_clean.csv",
                        help="Path to cleaned CSV (with churn column 0/1).")
    parser.add_argument("--train-indices", type=str, default="data/processed/train_indices.npy",
                        help="Numpy file with train indices (from preprocess step).")
    parser.add_argument("--mappings", type=str, default="models/binary_mappings.json",
                        help="Binary mappings json (from preprocess step).")
    parser.add_argument("--reports", type=str, default="reports", help="Reports directory.")
    parser.add_argument("--models", type=str, default="models", help="Models directory.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--splits", type=int, default=5, help="Stratified K-fold splits.")
    parser.add_argument("--topk", type=int, default=3, help="Top-k models to use in soft voting.")
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if "churn" not in df.columns:
        raise ValueError("Expected column 'churn' (0/1) in cleaned dataset.")

    y = df["churn"].astype(int)
    X = df.drop(columns=["churn"])

    # Apply saved binary mappings (LabelEncoding for binary columns)
    try:
        mappings = json.loads(Path(args.mappings).read_text(encoding="utf-8"))
        X = apply_binary_mappings(X, mappings)
    except FileNotFoundError:
        mappings = {}
        # Not fatal; we can still proceed with OneHot on those columns.

    train_idx = np.load(args.train_indices)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    preprocessor = build_preprocessor(X_train)

    cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    # -----------------------------
    # 1) Base models (ClassWeight vs SMOTE)
    # -----------------------------
    candidates = []

    # Logistic Regression
    candidates.append(("LogReg", LogisticRegression(max_iter=2000, random_state=args.seed)))
    # SVM (RBF)
    candidates.append(("SVM_RBF", SVC(kernel="rbf", probability=True, random_state=args.seed)))
    # KNN
    candidates.append(("KNN", KNeighborsClassifier()))
    # Random Forest
    candidates.append(("RandomForest", RandomForestClassifier(random_state=args.seed)))
    # Gradient boosting: XGBoost if available, else skip it (you can add LightGBM similarly)
    if HAS_XGB:
        candidates.append(("XGBoost", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )))

    rows = []

    def make_weighted_estimator(name, est):
        # Add class_weight='balanced' if the estimator supports it
        if hasattr(est, "get_params") and "class_weight" in est.get_params().keys():
            try:
                est = est.set_params(class_weight="balanced")
            except Exception:
                pass
        return est

    for name, est in candidates:
        # A) Class weights (where possible)
        est_w = make_weighted_estimator(name, est)
        pipe_w = Pipeline([("pre", preprocessor), ("clf", est_w)])
        metrics_w = cv_eval(pipe_w, X_train, y_train, cv, scorers)
        rows.append({"model": name, "imbalance": "class_weight", **metrics_w})

        # B) SMOTE (recommended; must be within CV to avoid leakage)
        if HAS_IMBLEARN:
            pipe_s = ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=args.seed)), ("clf", est)])
            metrics_s = cv_eval(pipe_s, X_train, y_train, cv, scorers)
            rows.append({"model": name, "imbalance": "smote", **metrics_s})
        else:
            rows.append({"model": name, "imbalance": "smote", "error": "install imbalanced-learn to run SMOTE"})

    results_df = pd.DataFrame(rows)
    results_path = tables_dir / "phase4_cv_baselines.csv"
    results_df.to_csv(results_path, index=False)

    # -----------------------------
    # 2) Hyperparameter tuning (at least 2 models)
    # -----------------------------
    tuned_rows = []

    # We'll tune LogReg and RandomForest on SMOTE path if available; else on class_weight.
    tune_use_smote = HAS_IMBLEARN

    # Logistic Regression search space
    lr = LogisticRegression(max_iter=3000, random_state=args.seed)
    if not tune_use_smote:
        lr = make_weighted_estimator("LogReg", lr)

    lr_pipe = (ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=args.seed)), ("clf", lr)])
               if tune_use_smote else
               Pipeline([("pre", preprocessor), ("clf", lr)]))

    lr_params = {
        "clf__C": np.logspace(-2, 2, 12),
        "clf__solver": ["liblinear", "lbfgs"],
    }

    lr_search = RandomizedSearchCV(
        lr_pipe,
        param_distributions=lr_params,
        n_iter=20,
        scoring="recall",
        cv=cv,
        random_state=args.seed,
        n_jobs=-1,
        verbose=0,
    )
    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    tuned_rows.append({
        "model": "LogReg_TUNED",
        "imbalance": "smote" if tune_use_smote else "class_weight",
        "best_params": json.dumps(lr_search.best_params_, ensure_ascii=False),
        **cv_eval(best_lr, X_train, y_train, cv, scorers),
    })

    # Random Forest search space
    rf = RandomForestClassifier(random_state=args.seed)
    if not tune_use_smote:
        rf = make_weighted_estimator("RandomForest", rf)

    rf_pipe = (ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=args.seed)), ("clf", rf)])
               if tune_use_smote else
               Pipeline([("pre", preprocessor), ("clf", rf)]))

    rf_params = {
        "clf__n_estimators": [200, 400, 700, 1000],
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__min_samples_split": [2, 5, 10],
    }

    rf_search = RandomizedSearchCV(
        rf_pipe,
        param_distributions=rf_params,
        n_iter=25,
        scoring="recall",
        cv=cv,
        random_state=args.seed,
        n_jobs=-1,
        verbose=0,
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    tuned_rows.append({
        "model": "RandomForest_TUNED",
        "imbalance": "smote" if tune_use_smote else "class_weight",
        "best_params": json.dumps(rf_search.best_params_, ensure_ascii=False),
        **cv_eval(best_rf, X_train, y_train, cv, scorers),
    })

    tuned_df = pd.DataFrame(tuned_rows)
    tuned_path = tables_dir / "phase4_cv_tuned.csv"
    tuned_df.to_csv(tuned_path, index=False)

    # -----------------------------
    # 3) Soft voting ensemble (top-k by recall_mean from tuned + baselines)
    # -----------------------------
    combined = pd.concat([
        results_df.drop(columns=[c for c in results_df.columns if c == "error"], errors="ignore"),
        tuned_df.drop(columns=["best_params"], errors="ignore"),
    ], ignore_index=True)

    combined = combined.dropna(subset=["recall_mean"]).sort_values("recall_mean", ascending=False)
    topk = combined.head(args.topk)

    # Map "model" to actual estimators for voting
    name_to_estimator = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=args.seed),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=args.seed),
        "KNN": KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(random_state=args.seed),
    }
    if HAS_XGB:
        name_to_estimator["XGBoost"] = XGBClassifier(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            random_state=args.seed, n_jobs=-1, eval_metric="logloss",
        )

    estimators = []
    for _, r in topk.iterrows():
        base_name = str(r["model"]).replace("_TUNED", "")
        if base_name not in name_to_estimator:
            continue
        est = name_to_estimator[base_name]
        imbalance = r.get("imbalance", "class_weight")
        if imbalance == "class_weight":
            est = make_weighted_estimator(base_name, est)
            pipe = Pipeline([("pre", preprocessor), ("clf", est)])
        else:
            if not HAS_IMBLEARN:
                continue
            pipe = ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=args.seed)), ("clf", est)])
        estimators.append((f"{base_name}_{imbalance}", pipe))

    if len(estimators) >= 2:
        voting_estimators = [(f"{name}_{i}", est) for i, (name, est) in enumerate(estimators)]
        voting = VotingClassifier(
            estimators=voting_estimators,
            voting="soft",
            n_jobs=-1
        )
        voting_metrics = cv_eval(voting, X_train, y_train, cv, scorers)
        voting_row = {"model": f"VotingSoft_top{len(estimators)}", "imbalance": "mixed", **voting_metrics}
        combined_out = pd.concat([combined, pd.DataFrame([voting_row])], ignore_index=True)
    else:
        combined_out = combined.copy()

    combined_path = tables_dir / "phase4_cv_all_results.csv"
    combined_out.to_csv(combined_path, index=False)

    # Pick the best by recall_mean (tie-breaker: f1_mean)
    best = combined_out.sort_values(["recall_mean", "f1_mean"], ascending=False).iloc[0]
    best_model_name = best["model"]

    # Fit best model on full train and save it (for Phase 5 evaluation)
    best_pipeline = None
    if best_model_name == "LogReg_TUNED":
        best_pipeline = best_lr
    elif best_model_name == "RandomForest_TUNED":
        best_pipeline = best_rf
    elif best_model_name.startswith("VotingSoft_") and len(estimators) >= 2:
        best_pipeline = voting
    else:
        # fallback: rebuild from candidates + imbalance
        base_name = str(best_model_name)
        imbalance = str(best.get("imbalance", "class_weight"))
        if base_name in name_to_estimator:
            est = name_to_estimator[base_name]
            if imbalance == "class_weight":
                est = make_weighted_estimator(base_name, est)
                best_pipeline = Pipeline([("pre", preprocessor), ("clf", est)])
            elif imbalance == "smote" and HAS_IMBLEARN:
                best_pipeline = ImbPipeline([("pre", preprocessor), ("smote", SMOTE(random_state=args.seed)), ("clf", est)])

    if best_pipeline is not None:
        best_pipeline.fit(X_train, y_train)
        dump(best_pipeline, models_dir / "phase4_best_model.joblib")

    # -----------------------------
    # 4) Write a short markdown report
    # -----------------------------
    lines = []
    lines.append("# Phase 4 — Modeling & Optimization (Project 7)")
    lines.append("")
    lines.append("## What we did")
    lines.append("- Compared **class_weight** vs **SMOTE** for imbalance handling (SMOTE is applied inside CV).")
    lines.append("- Trained baseline models: Logistic Regression, SVM (RBF), KNN, Random Forest, and (optionally) XGBoost.")
    lines.append("- Hyperparameter tuning for **2 models** (LogReg + RandomForest) using RandomizedSearchCV (optimize Recall).")
    lines.append(f"- Advanced validation: StratifiedKFold (splits={args.splits}) and we report mean±std for Recall.")
    lines.append("")
    lines.append("## Key tables")
    lines.append(f"- Baselines: `{results_path.as_posix()}`")
    lines.append(f"- Tuned: `{tuned_path.as_posix()}`")
    lines.append(f"- Combined + voting: `{combined_path.as_posix()}`")
    lines.append("")
    lines.append("## Best model (by Recall, tie-breaker F1)")
    lines.append(f"- **{best_model_name}**")
    lines.append("")
    lines.append("## Notes")
    if not HAS_IMBLEARN:
        lines.append("- SMOTE was skipped because `imbalanced-learn` is not installed. Install it to complete the SMOTE requirement.")
    if not HAS_XGB:
        lines.append("- XGBoost was skipped because `xgboost` is not installed. Install it to match the recommended Gradient Boosting model.")
    lines.append("")
    lines.append("## Snapshot (top rows by recall_mean)")
    top_show = combined_out.sort_values("recall_mean", ascending=False).head(8).copy()
    cols = [c for c in ["model", "imbalance", "recall_mean", "recall_std", "f1_mean", "precision_mean", "accuracy_mean"] if c in top_show.columns]
    lines.append(markdown_table(top_show[cols].round(4)))

    (reports_dir / "modeling.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
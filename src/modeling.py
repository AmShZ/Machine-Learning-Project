import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.pipeline_utils import (
    FeatureSubsetTransformer,
    add_engineered_features,
    apply_binary_mappings,
    build_preprocessor,
    load_binary_mappings,
    load_selected_features,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except Exception:
    HAS_IMBLEARN = False


RANDOM_SEED = 42


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def cv_eval(pipeline, X: pd.DataFrame, y: pd.Series, cv, scorers: dict[str, object]) -> dict[str, float]:
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scorers,
        n_jobs=-1,
        error_score="raise",
        return_train_score=False,
    )
    metrics: dict[str, float] = {}
    for key, values in result.items():
        if key.startswith("test_"):
            metric = key.replace("test_", "")
            metrics[f"{metric}_mean"] = float(np.mean(values))
            metrics[f"{metric}_std"] = float(np.std(values))
    return metrics


def make_weighted_estimator(estimator):
    if hasattr(estimator, "get_params") and "class_weight" in estimator.get_params().keys():
        try:
            estimator = estimator.set_params(class_weight="balanced")
        except Exception:
            return estimator
    return estimator


def build_pipeline(estimator, preprocessor, selected_features: list[str], use_smote: bool, seed: int):
    selector = FeatureSubsetTransformer(selected_features=selected_features or None)
    if use_smote and HAS_IMBLEARN:
        return ImbPipeline(
            [
                ("pre", preprocessor),
                ("select", selector),
                ("smote", SMOTE(random_state=seed)),
                ("clf", estimator),
            ]
        )
    return Pipeline(
        [
            ("pre", preprocessor),
            ("select", selector),
            ("clf", estimator),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Modeling and Optimization")
    parser.add_argument("--data", type=str, default="data/processed/telco_churn_clean.csv")
    parser.add_argument("--train-indices", type=str, default="data/processed/train_indices.npy")
    parser.add_argument("--mappings", type=str, default="models/binary_mappings.json")
    parser.add_argument("--selected-features", type=str, default="reports/tables/selected_features_final.csv")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--models", type=str, default="models")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    tables_dir = reports_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(args.models)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if "churn" not in df.columns:
        raise ValueError("Expected `churn` column in cleaned dataset.")

    df = add_engineered_features(df)

    y = df["churn"].astype(int).reset_index(drop=True)
    X = df.drop(columns=["churn"])

    mappings = load_binary_mappings(args.mappings)
    if mappings:
        X = apply_binary_mappings(X, mappings)

    train_idx = np.load(args.train_indices)
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)

    selected_features = load_selected_features(args.selected_features)

    preprocessor, _, _ = build_preprocessor(X_train)

    cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=args.seed)

    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=1, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=1, zero_division=0),
    }

    model_builders: dict[str, object] = {
        "LogReg": lambda: LogisticRegression(max_iter=2000, random_state=args.seed),
        "SVM_RBF": lambda: SVC(kernel="rbf", probability=True, random_state=args.seed),
        "KNN": lambda: KNeighborsClassifier(),
        "RandomForest": lambda: RandomForestClassifier(random_state=args.seed),
    }
    if HAS_XGB:
        model_builders["XGBoost"] = lambda: XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=args.seed,
            n_jobs=-1,
            eval_metric="logloss",
        )

    baseline_rows: list[dict[str, object]] = []

    for model_name, builder in model_builders.items():
        weighted_estimator = make_weighted_estimator(builder())
        weighted_pipeline = build_pipeline(weighted_estimator, preprocessor, selected_features, use_smote=False, seed=args.seed)
        weighted_metrics = cv_eval(weighted_pipeline, X_train, y_train, cv, scorers)
        baseline_rows.append({"model": model_name, "imbalance": "class_weight", "status": "ok", **weighted_metrics})

        if HAS_IMBLEARN:
            smote_pipeline = build_pipeline(builder(), preprocessor, selected_features, use_smote=True, seed=args.seed)
            smote_metrics = cv_eval(smote_pipeline, X_train, y_train, cv, scorers)
            baseline_rows.append({"model": model_name, "imbalance": "smote", "status": "ok", **smote_metrics})
        else:
            baseline_rows.append(
                {
                    "model": model_name,
                    "imbalance": "smote",
                    "status": "skipped_missing_imbalanced_learn",
                    "accuracy_mean": np.nan,
                    "accuracy_std": np.nan,
                    "precision_mean": np.nan,
                    "precision_std": np.nan,
                    "recall_mean": np.nan,
                    "recall_std": np.nan,
                    "f1_mean": np.nan,
                    "f1_std": np.nan,
                }
            )

    baselines_df = pd.DataFrame(baseline_rows)
    baselines_path = tables_dir / "phase4_cv_baselines.csv"
    baselines_df.to_csv(baselines_path, index=False)

    tune_with_smote = HAS_IMBLEARN
    tuned_rows: list[dict[str, object]] = []

    lr_estimator = LogisticRegression(max_iter=3000, random_state=args.seed)
    if not tune_with_smote:
        lr_estimator = make_weighted_estimator(lr_estimator)
    lr_pipeline = build_pipeline(lr_estimator, preprocessor, selected_features, use_smote=tune_with_smote, seed=args.seed)
    lr_params = {
        "clf__C": np.logspace(-2, 2, 12),
        "clf__solver": ["liblinear", "lbfgs"],
    }

    lr_search = RandomizedSearchCV(
        lr_pipeline,
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
    tuned_rows.append(
        {
            "model": "LogReg_TUNED",
            "imbalance": "smote" if tune_with_smote else "class_weight",
            "best_params": json.dumps(lr_search.best_params_),
            "status": "ok",
            **cv_eval(best_lr, X_train, y_train, cv, scorers),
        }
    )

    rf_estimator = RandomForestClassifier(random_state=args.seed)
    if not tune_with_smote:
        rf_estimator = make_weighted_estimator(rf_estimator)
    rf_pipeline = build_pipeline(rf_estimator, preprocessor, selected_features, use_smote=tune_with_smote, seed=args.seed)
    rf_params = {
        "clf__n_estimators": [200, 400, 700, 1000],
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_leaf": [1, 2, 5, 10],
        "clf__min_samples_split": [2, 5, 10],
    }

    rf_search = RandomizedSearchCV(
        rf_pipeline,
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
    tuned_rows.append(
        {
            "model": "RandomForest_TUNED",
            "imbalance": "smote" if tune_with_smote else "class_weight",
            "best_params": json.dumps(rf_search.best_params_),
            "status": "ok",
            **cv_eval(best_rf, X_train, y_train, cv, scorers),
        }
    )

    tuned_df = pd.DataFrame(tuned_rows)
    tuned_path = tables_dir / "phase4_cv_tuned.csv"
    tuned_df.to_csv(tuned_path, index=False)

    metric_cols = [
        "model",
        "imbalance",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "status",
    ]

    baseline_valid = baselines_df.dropna(subset=["recall_mean"])[metric_cols]
    tuned_valid = tuned_df.dropna(subset=["recall_mean"])[metric_cols]
    combined_df = pd.concat([baseline_valid, tuned_valid], ignore_index=True)
    combined_df = combined_df.sort_values(["recall_mean", "f1_mean"], ascending=False).reset_index(drop=True)

    def pipeline_for_result(model_name: str, imbalance: str):
        if model_name == "LogReg_TUNED":
            return best_lr
        if model_name == "RandomForest_TUNED":
            return best_rf
        if model_name not in model_builders:
            return None
        estimator = model_builders[model_name]()
        if imbalance == "class_weight":
            estimator = make_weighted_estimator(estimator)
            return build_pipeline(estimator, preprocessor, selected_features, use_smote=False, seed=args.seed)
        if imbalance == "smote" and HAS_IMBLEARN:
            return build_pipeline(estimator, preprocessor, selected_features, use_smote=True, seed=args.seed)
        return None

    top_rows = combined_df.head(max(args.topk, 1))
    voting = None
    if len(top_rows) >= 2:
        voting_estimators = []
        for idx, row in top_rows.iterrows():
            candidate = pipeline_for_result(str(row["model"]), str(row["imbalance"]))
            if candidate is not None:
                voting_estimators.append((f"m{idx}", candidate))
        if len(voting_estimators) >= 2:
            voting = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)
            voting_metrics = cv_eval(voting, X_train, y_train, cv, scorers)
            voting_row = {
                "model": f"VotingSoft_top{len(voting_estimators)}",
                "imbalance": "mixed",
                "status": "ok",
                **voting_metrics,
            }
            combined_df = pd.concat([combined_df, pd.DataFrame([voting_row])], ignore_index=True)

    combined_df = combined_df.sort_values(["recall_mean", "f1_mean"], ascending=False).reset_index(drop=True)
    combined_path = tables_dir / "phase4_cv_all_results.csv"
    combined_df.to_csv(combined_path, index=False)

    best_row = combined_df.iloc[0]
    best_model_name = str(best_row["model"])
    best_imbalance = str(best_row["imbalance"])

    if best_model_name.startswith("VotingSoft") and voting is not None:
        best_pipeline = voting
    else:
        best_pipeline = pipeline_for_result(best_model_name, best_imbalance)

    if best_pipeline is not None:
        best_pipeline.fit(X_train, y_train)
        dump(best_pipeline, models_dir / "phase4_best_model.joblib")

    top_preview = combined_df.sort_values(["recall_mean", "f1_mean"], ascending=False).head(8).copy()
    preview_cols = [
        "model",
        "imbalance",
        "recall_mean",
        "recall_std",
        "precision_mean",
        "f1_mean",
        "accuracy_mean",
    ]
    top_preview = top_preview[preview_cols].round(4)

    lines: list[str] = []
    lines.append("# Phase 4 - Modeling and Optimization")
    lines.append("")
    lines.append("## Setup")
    lines.append(f"- StratifiedKFold splits: {args.splits}")
    lines.append(f"- Primary metric: recall for churn=1")
    lines.append(f"- Feature engineering from Phase 3 applied: yes")
    lines.append(f"- Selected post-onehot feature subset used: {'yes' if len(selected_features) > 0 else 'no'}")
    if len(selected_features) > 0:
        lines.append(f"- Selected feature count: {len(selected_features)}")
        lines.append(f"- Selected feature source: `{Path(args.selected_features).as_posix()}`")
    lines.append("")
    lines.append("## Imbalance handling")
    lines.append("- class_weight=balanced")
    lines.append("- SMOTE inside CV pipeline")
    lines.append("")
    lines.append("## Models")
    lines.append("- Logistic Regression")
    lines.append("- SVM (RBF)")
    lines.append("- KNN")
    lines.append("- Random Forest")
    lines.append("- Gradient boosting: XGBoost when installed")
    lines.append("")
    lines.append("## Hyperparameter tuning")
    lines.append("- Tuned models: Logistic Regression and Random Forest")
    lines.append("- Search method: RandomizedSearchCV optimized for recall")
    lines.append("")
    lines.append("## Best model")
    lines.append(f"- {best_model_name} ({best_imbalance})")
    lines.append("")
    lines.append("## Output tables")
    lines.append(f"- `{baselines_path.as_posix()}`")
    lines.append(f"- `{tuned_path.as_posix()}`")
    lines.append(f"- `{combined_path.as_posix()}`")
    lines.append("")
    lines.append("## Top CV results")
    lines.append(markdown_table(top_preview))
    lines.append("")
    lines.append("## Notes")
    if not HAS_IMBLEARN:
        lines.append("- `imbalanced-learn` not installed; SMOTE rows were skipped.")
    if not HAS_XGB:
        lines.append("- `xgboost` not installed; gradient boosting baseline was skipped.")
    if HAS_IMBLEARN and HAS_XGB:
        lines.append("- All required optional components were available.")

    (reports_dir / "modeling.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

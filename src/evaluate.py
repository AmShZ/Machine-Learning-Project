import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.pipeline_utils import add_engineered_features, apply_binary_mappings, load_binary_mappings

matplotlib.use("Agg")


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def make_metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, threshold: float | None) -> dict[str, object]:
    return {
        "scenario": name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def report_table(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    table = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    table.insert(0, "scenario", name)
    return table


def save_confusion_figure(matrix: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_threshold_figure(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["threshold"], df["precision"], marker="o", label="precision")
    ax.plot(df["threshold"], df["recall"], marker="o", label="recall")
    ax.plot(df["threshold"], df["f1"], marker="o", label="f1")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title("Threshold sweep on held-out test set")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def minmax_scale(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    low = float(np.min(arr))
    high = float(np.max(arr))
    if high - low <= 0:
        return np.zeros_like(arr, dtype=float)
    return (arr - low) / (high - low)


def get_scores_for_roc(model, X: pd.DataFrame) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                scores = proba[:, 1].astype(float)
                return scores, scores, "ok"
        except Exception:
            pass
    if hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)
            decision = np.asarray(decision)
            if decision.ndim == 2:
                if decision.shape[1] >= 2:
                    decision = decision[:, 1]
                else:
                    decision = decision.ravel()
            raw = decision.astype(float)
            return raw, minmax_scale(raw), "ok"
        except Exception:
            pass
    return None, None, "no_score"


def get_pipeline_feature_names(estimator) -> np.ndarray | None:
    if not hasattr(estimator, "named_steps"):
        return None
    steps = estimator.named_steps
    if "pre" not in steps:
        return None
    pre = steps["pre"]
    if not hasattr(pre, "get_feature_names_out"):
        return None
    names = np.asarray(pre.get_feature_names_out(), dtype=object)
    if "select" in steps and hasattr(steps["select"], "get_feature_names_out"):
        try:
            names = np.asarray(steps["select"].get_feature_names_out(names), dtype=object)
        except Exception:
            pass
    return names


def get_estimator_importance_series(estimator) -> pd.Series | None:
    model = estimator
    feature_names = None
    if hasattr(estimator, "named_steps"):
        feature_names = get_pipeline_feature_names(estimator)
        if "clf" in estimator.named_steps:
            model = estimator.named_steps["clf"]
    importances = None
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 1:
            importances = np.abs(coef)
        elif coef.shape[0] >= 1:
            importances = np.abs(coef[0])
    elif hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    if importances is None:
        return None
    values = np.asarray(importances, dtype=float).ravel()
    if feature_names is None:
        feature_names = np.asarray([f"feature_{idx}" for idx in range(len(values))], dtype=object)
    if len(feature_names) != len(values):
        n = min(len(feature_names), len(values))
        feature_names = feature_names[:n]
        values = values[:n]
    return pd.Series(np.abs(values), index=feature_names.astype(str))


def get_top_features(best_model, X_ref: pd.DataFrame, y_ref: np.ndarray, top_n: int = 15) -> pd.DataFrame:
    importance_series = None
    if isinstance(best_model, VotingClassifier):
        member_series: list[pd.Series] = []
        members = list(getattr(best_model, "estimators_", []))
        for member in members:
            series = get_estimator_importance_series(member)
            if series is None or len(series) == 0:
                continue
            total = float(series.sum())
            if total > 0:
                member_series.append(series / total)
        if member_series:
            merged = pd.concat(member_series, axis=1).fillna(0.0)
            importance_series = merged.mean(axis=1)
    else:
        series = get_estimator_importance_series(best_model)
        if series is not None and len(series) > 0:
            importance_series = series
    if importance_series is None or len(importance_series) == 0:
        sample_size = min(len(X_ref), 600)
        X_sample = X_ref.iloc[:sample_size].copy()
        y_sample = y_ref[:sample_size]
        perm = permutation_importance(
            best_model,
            X_sample,
            y_sample,
            scoring="recall",
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        importance_series = pd.Series(np.abs(perm.importances_mean), index=X_sample.columns.astype(str))
    top = importance_series.sort_values(ascending=False).head(top_n)
    return pd.DataFrame({"feature": top.index.astype(str), "importance": top.values.astype(float)})


def build_business_recommendations(top_features: list[str]) -> list[str]:
    recommendations: list[str] = []

    def has(token: str) -> bool:
        token_l = token.lower()
        return any(token_l in feature.lower() for feature in top_features)

    if has("Contract"):
        recommendations.append("- Prioritize month-to-month customers with renewal incentives and migration paths to longer contracts.")
    if has("OnlineSecurity") or has("TechSupport") or has("OnlineBackup"):
        recommendations.append("- Bundle security, backup, and support services for high-risk accounts to reduce service-related churn.")
    if has("tenure") or has("TenureGroup"):
        recommendations.append("- Launch a first-year retention program with proactive outreach during early-tenure periods.")
    if has("PaymentMethod"):
        recommendations.append("- Target electronic-check customers with autopay conversion campaigns and billing support.")
    if has("MonthlyCharges") or has("AvgChargesPerMonth") or has("TotalCharges"):
        recommendations.append("- Use personalized pricing reviews for customers with high monthly or per-month charge intensity.")

    idx = 0
    while len(recommendations) < 4 and idx < len(top_features):
        recommendations.append(f"- Monitor churn risk alerts triggered by `{top_features[idx]}` and route cases to retention specialists.")
        idx += 1

    return recommendations[:6]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Final evaluation and reporting")
    parser.add_argument("--data", type=str, default="data/processed/telco_churn_clean.csv")
    parser.add_argument("--train-indices", type=str, default="data/processed/train_indices.npy")
    parser.add_argument("--test-indices", type=str, default="data/processed/test_indices.npy")
    parser.add_argument("--mappings", type=str, default="models/binary_mappings.json")
    parser.add_argument("--model", type=str, default="models/phase4_best_model.joblib")
    parser.add_argument("--top4-models", type=str, default="models/phase4_top4_models.joblib")
    parser.add_argument("--top4-metadata", type=str, default="reports/tables/phase4_top4_models.csv")
    parser.add_argument("--reports", type=str, default="reports")
    parser.add_argument("--threshold_start", type=float, default=0.1)
    parser.add_argument("--threshold_end", type=float, default=0.9)
    parser.add_argument("--threshold_step", type=float, default=0.1)
    args = parser.parse_args()

    reports_dir = Path(args.reports)
    tables_dir = reports_dir / "tables"
    figures_dir = reports_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    if "churn" not in df.columns:
        raise ValueError("Expected `churn` column in cleaned dataset.")

    df = add_engineered_features(df)
    y = df["churn"].astype(int).to_numpy()
    X = df.drop(columns=["churn"])

    mappings = load_binary_mappings(args.mappings)
    if mappings:
        X = apply_binary_mappings(X, mappings)

    train_idx = np.load(args.train_indices)
    test_idx = np.load(args.test_indices)

    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y[train_idx]
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]

    model = load(args.model)

    top4_models_path = Path(args.top4_models)
    if not top4_models_path.exists():
        raise FileNotFoundError(
            f"Missing {top4_models_path.as_posix()}. Run Phase 4 first: python -m src.modeling"
        )
    top4_metadata_path = Path(args.top4_metadata)
    if not top4_metadata_path.exists():
        raise FileNotFoundError(
            f"Missing {top4_metadata_path.as_posix()}. Run Phase 4 first: python -m src.modeling"
        )

    top4_models = load(top4_models_path)
    top4_meta = pd.read_csv(top4_metadata_path)
    if "rank" not in top4_meta.columns:
        raise ValueError("Top-4 metadata must include a `rank` column.")

    y_pred_default = model.predict(X_test)

    metrics_rows: list[dict[str, object]] = []
    metrics_rows.append(make_metric_row("default_predict", y_test, y_pred_default, threshold=0.5))

    default_confusion = confusion_matrix(y_test, y_pred_default, labels=[0, 1])
    confusion_rows = [
        {
            "scenario": "default_predict",
            "threshold": 0.5,
            "tn": int(default_confusion[0, 0]),
            "fp": int(default_confusion[0, 1]),
            "fn": int(default_confusion[1, 0]),
            "tp": int(default_confusion[1, 1]),
        }
    ]

    class_report_frames = [report_table("default_predict", y_test, y_pred_default)]

    threshold_sweep = pd.DataFrame()
    threshold_note = "Threshold analysis not available because probability scores were not exposed by the model."

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if np.asarray(proba).ndim == 2 and np.asarray(proba).shape[1] >= 2:
            scores = np.asarray(proba)[:, 1]
            thresholds = np.arange(args.threshold_start, args.threshold_end + 1e-9, args.threshold_step)
            sweep_rows = []
            for threshold in thresholds:
                y_pred_threshold = (scores >= threshold).astype(int)
                sweep_rows.append(make_metric_row("threshold_sweep", y_test, y_pred_threshold, threshold=float(np.round(threshold, 4))))
            threshold_sweep = pd.DataFrame(sweep_rows)
            default_precision = float(metrics_rows[0]["precision"])
            precision_floor = max(0.0, default_precision - 0.05)
            eligible = threshold_sweep[threshold_sweep["precision"] >= precision_floor]
            if eligible.empty:
                chosen = threshold_sweep.iloc[(threshold_sweep["threshold"] - 0.5).abs().argmin()]
                threshold_note = (
                    "No threshold met the precision floor of "
                    f"{precision_floor:.3f}. Using threshold {float(chosen['threshold']):.2f}."
                )
            else:
                chosen = eligible.sort_values(["recall", "f1", "precision"], ascending=False).iloc[0]
                threshold_note = (
                    "Selected threshold "
                    f"{float(chosen['threshold']):.2f} by maximizing recall while keeping precision >= {precision_floor:.3f}."
                )
            selected_threshold = float(chosen["threshold"])
            y_pred_selected = (scores >= selected_threshold).astype(int)
            metrics_rows.append(make_metric_row("selected_threshold", y_test, y_pred_selected, threshold=selected_threshold))
            selected_confusion = confusion_matrix(y_test, y_pred_selected, labels=[0, 1])
            confusion_rows.append(
                {
                    "scenario": "selected_threshold",
                    "threshold": selected_threshold,
                    "tn": int(selected_confusion[0, 0]),
                    "fp": int(selected_confusion[0, 1]),
                    "fn": int(selected_confusion[1, 0]),
                    "tp": int(selected_confusion[1, 1]),
                }
            )
            class_report_frames.append(report_table("selected_threshold", y_test, y_pred_selected))
            save_confusion_figure(
                selected_confusion,
                figures_dir / "phase5_confusion_matrix.png",
                f"Confusion matrix at threshold {selected_threshold:.2f}",
            )
            save_threshold_figure(threshold_sweep, figures_dir / "phase5_threshold_tradeoff.png")

    metrics_df = pd.DataFrame(metrics_rows)
    confusion_df = pd.DataFrame(confusion_rows)
    class_report_df = pd.concat(class_report_frames, ignore_index=True)

    threshold_path = tables_dir / "phase5_threshold_sweep.csv"
    if threshold_sweep.empty:
        pd.DataFrame(
            [{"status": "not_available", "reason": "model does not expose predict_proba for threshold sweep"}]
        ).to_csv(threshold_path, index=False)
    else:
        threshold_sweep.to_csv(threshold_path, index=False)

    metrics_path = tables_dir / "phase5_test_metrics.csv"
    confusion_path = tables_dir / "phase5_confusion_matrix.csv"
    class_report_path = tables_dir / "phase5_classification_report.csv"

    metrics_df.to_csv(metrics_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)
    class_report_df.to_csv(class_report_path, index=False)

    save_confusion_figure(default_confusion, figures_dir / "phase5_confusion_matrix_default.png", "Confusion matrix at default predict")

    auc_rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_curves = 0
    top4_meta = top4_meta.sort_values("rank").reset_index(drop=True)
    for _, row in top4_meta.iterrows():
        rank = int(row["rank"])
        model_id = f"rank{rank}"
        model_name = str(row.get("model", "unknown"))
        imbalance = str(row.get("imbalance", "unknown"))
        candidate = top4_models.get(model_id)
        if candidate is None:
            auc_rows.append(
                {
                    "model_id": model_id,
                    "rank": rank,
                    "model": model_name,
                    "imbalance": imbalance,
                    "auc_roc": np.nan,
                    "status": "missing_model",
                }
            )
            continue
        raw_scores, plot_scores, score_status = get_scores_for_roc(candidate, X_test)
        if score_status != "ok" or raw_scores is None or plot_scores is None:
            auc_rows.append(
                {
                    "model_id": model_id,
                    "rank": rank,
                    "model": model_name,
                    "imbalance": imbalance,
                    "auc_roc": np.nan,
                    "status": "no_score",
                }
            )
            continue
        try:
            auc_value = float(roc_auc_score(y_test, raw_scores))
            fpr, tpr, _ = roc_curve(y_test, plot_scores)
            ax.plot(fpr, tpr, linewidth=2, label=f"{model_id} {model_name} ({imbalance}) AUC={auc_value:.3f}")
            plotted_curves += 1
            auc_rows.append(
                {
                    "model_id": model_id,
                    "rank": rank,
                    "model": model_name,
                    "imbalance": imbalance,
                    "auc_roc": auc_value,
                    "status": "ok",
                }
            )
        except Exception:
            auc_rows.append(
                {
                    "model_id": model_id,
                    "rank": rank,
                    "model": model_name,
                    "imbalance": imbalance,
                    "auc_roc": np.nan,
                    "status": "score_error",
                }
            )

    if plotted_curves > 0:
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC curves for top 4 Phase 4 models")
        ax.legend(loc="lower right", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        ax.set_title("ROC curves for top 4 Phase 4 models")
        ax.text(0.5, 0.5, "No ROC curves available", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(figures_dir / "phase5_roc_curves.png", dpi=160)
    plt.close(fig)

    auc_df = pd.DataFrame(auc_rows, columns=["model_id", "rank", "model", "imbalance", "auc_roc", "status"])
    auc_df.to_csv(tables_dir / "phase5_auc_roc.csv", index=False)

    top_features_df = get_top_features(model, X_train, y_train, top_n=15)
    top_features_df.to_csv(tables_dir / "phase5_top_features.csv", index=False)

    phase4_table_path = reports_dir / "tables" / "phase4_cv_all_results.csv"
    phase4_summary = "Not available"
    if phase4_table_path.exists():
        phase4_table = pd.read_csv(phase4_table_path)
        if len(phase4_table) > 0 and "recall_mean" in phase4_table.columns:
            best_phase4 = phase4_table.sort_values(["recall_mean", "f1_mean"], ascending=False).iloc[0]
            phase4_summary = (
                f"{best_phase4['model']} ({best_phase4['imbalance']}), "
                f"CV recall={best_phase4['recall_mean']:.4f}, CV f1={best_phase4['f1_mean']:.4f}"
            )

    lines: list[str] = []
    lines.append("# Final Report - Telco Customer Churn")
    lines.append("")
    lines.append("## Project summary")
    lines.append("- Primary objective: maximize recall for churn class (1) with supporting precision, F1, and accuracy.")
    lines.append("- Data split policy: fixed stratified split from Phase 2 with seed 42 and reused in Phases 3-5.")
    lines.append("")
    lines.append("## Phase outputs")
    lines.append("- Phase 1: `reports/eda.md` and figures in `reports/figures/`")
    lines.append("- Phase 2: `reports/preprocessing.md` and processed datasets in `data/processed/`")
    lines.append("- Phase 3: `reports/feature_engineering.md` and selected feature tables in `reports/tables/`")
    lines.append("- Phase 4: `reports/modeling.md`, CV tables, and `models/phase4_best_model.joblib`")
    lines.append("- Phase 5: test evaluation tables and this report")
    lines.append("")
    lines.append("## Phase 4 best CV result")
    lines.append(f"- {phase4_summary}")
    lines.append("")
    lines.append("## Phase 5 held-out test metrics")
    lines.append(markdown_table(metrics_df.round(4)))
    lines.append("")
    lines.append("## Confusion matrix summary")
    lines.append(markdown_table(confusion_df))
    lines.append("")
    lines.append("## Classification report")
    report_preview = class_report_df.copy()
    numeric_cols = [col for col in ["precision", "recall", "f1-score", "support"] if col in report_preview.columns]
    for col in numeric_cols:
        report_preview[col] = report_preview[col].astype(float).round(4)
    lines.append(markdown_table(report_preview))
    lines.append("")
    lines.append("## Threshold discussion")
    lines.append(f"- {threshold_note}")
    if not threshold_sweep.empty:
        lines.append("- Threshold sweep table: `reports/tables/phase5_threshold_sweep.csv`")
        lines.append("- Threshold tradeoff figure: `reports/figures/phase5_threshold_tradeoff.png`")
    lines.append("")
    lines.append("## Key artifacts")
    lines.append("- `reports/tables/phase5_test_metrics.csv`")
    lines.append("- `reports/tables/phase5_confusion_matrix.csv`")
    lines.append("- `reports/tables/phase5_classification_report.csv`")
    lines.append("- `reports/final_report.md`")
    lines.append("")
    lines.append("## ROC-AUC comparison (Top 4 models)")
    lines.append("")
    lines.append("Table: `tables/phase5_auc_roc.csv`")
    lines.append("")
    lines.append("![ROC curves](figures/phase5_roc_curves.png)")
    lines.append("")
    auc_preview = auc_df.copy()
    if "auc_roc" in auc_preview.columns:
        auc_preview["auc_roc"] = auc_preview["auc_roc"].astype(float).round(4)
    lines.append(markdown_table(auc_preview))
    lines.append("")
    lines.append("## Winner model and most important features")
    lines.append("")
    lines.append("Winner model artifact: `models/phase4_best_model.joblib`")
    lines.append("")
    top_feature_names = top_features_df["feature"].head(5).tolist()
    if top_feature_names:
        lines.append(f"Highest-impact features: {', '.join(top_feature_names)}")
        lines.append("")
    lines.append("Feature table: `tables/phase5_top_features.csv`")
    lines.append("")
    lines.append(markdown_table(top_features_df.round(6)))
    lines.append("")
    lines.append("## Business recommendations")
    for rec in build_business_recommendations(top_features_df["feature"].astype(str).tolist()):
        lines.append(rec)

    (reports_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

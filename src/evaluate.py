import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Final evaluation and reporting")
    parser.add_argument("--data", type=str, default="data/processed/telco_churn_clean.csv")
    parser.add_argument("--train-indices", type=str, default="data/processed/train_indices.npy")
    parser.add_argument("--test-indices", type=str, default="data/processed/test_indices.npy")
    parser.add_argument("--mappings", type=str, default="models/binary_mappings.json")
    parser.add_argument("--model", type=str, default="models/phase4_best_model.joblib")
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

    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y[test_idx]

    model = load(args.model)

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
    selected_threshold = 0.5
    threshold_note = "Threshold analysis not available because probability scores were not exposed by the model."

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            scores = proba[:, 1]
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

    (reports_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

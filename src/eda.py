import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.pipeline_utils import resolve_raw_data_path

matplotlib.use("Agg")


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def save_hist(series: pd.Series, out_path: Path, title: str, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna().to_numpy(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_bar_counts(series: pd.Series, out_path: Path, title: str) -> None:
    counts = series.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index.astype(str), counts.to_numpy())
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_churn_rate(df: pd.DataFrame, column: str, out_path: Path) -> pd.DataFrame:
    tmp = df.copy()
    tmp["churn"] = (tmp["Churn"] == "Yes").astype(int)
    rates = tmp.groupby(column, dropna=False)["churn"].mean().sort_values(ascending=False)
    table = rates.reset_index().rename(columns={"churn": "churn_rate"})
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(table[column].astype(str), table["churn_rate"].to_numpy())
    ax.set_title(f"Churn rate by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("churn rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return table


def save_boxplot_by_target(df: pd.DataFrame, feature: str, out_path: Path) -> None:
    yes_values = df.loc[df["Churn"] == "Yes", feature].dropna().to_numpy()
    no_values = df.loc[df["Churn"] == "No", feature].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    try:
        ax.boxplot([no_values, yes_values], tick_labels=["No", "Yes"], showfliers=False)
    except TypeError:
        ax.boxplot([no_values, yes_values], labels=["No", "Yes"], showfliers=False)
    ax.set_title(f"{feature} vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(feature)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_corr_heatmap(df_num: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    corr = df_num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    image = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for row_idx in range(corr.shape[0]):
        for col_idx in range(corr.shape[1]):
            ax.text(col_idx, row_idx, f"{corr.iat[row_idx, col_idx]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap (numeric)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return corr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--reports", type=str, default="reports")
    args = parser.parse_args()

    data_path = resolve_raw_data_path(args.data)
    reports_dir = Path(args.reports)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    shape = df.shape
    duplicate_rows = int(df.duplicated().sum())
    duplicate_customer_ids = int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else 0

    dtypes_table = (
        df.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    missing_table = (
        df.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .query("missing > 0")
        .reset_index()
        .rename(columns={"index": "column"})
    )

    total_charges_blank = 0
    if "TotalCharges" in df.columns:
        total_charges_blank = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    else:
        df["TotalCharges_num"] = np.nan

    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num"]
    numeric_cols = [column for column in numeric_cols if column in df.columns]
    df_num = df[numeric_cols].copy()

    numeric_summary = df_num.describe().T.reset_index().rename(columns={"index": "feature"})
    numeric_summary["missing"] = df_num.isna().sum().to_numpy()

    churn_rate = float((df["Churn"] == "Yes").mean())

    save_bar_counts(df["Churn"], figures_dir / "target_churn.png", "Target distribution (Churn)")

    for column in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if column in df.columns:
            save_hist(df[column], figures_dir / f"hist_{column}.png", f"Histogram: {column}")

    cat_cols = [column for column in df.columns if df[column].dtype == "object" and column not in {"customerID", "Churn", "TotalCharges"}]
    for column in cat_cols:
        save_bar_counts(df[column], figures_dir / f"cat_{column}.png", f"Counts: {column}")

    save_corr_heatmap(df_num, figures_dir / "corr_heatmap.png")

    for column in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if column in df.columns:
            save_boxplot_by_target(df, column, figures_dir / f"box_{column}_by_churn.png")

    key_cols = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    churn_rate_tables: dict[str, pd.DataFrame] = {}
    for column in key_cols:
        if column in df.columns:
            churn_rate_tables[column] = save_churn_rate(df, column, figures_dir / f"churnrate_{column}.png")

    insights: list[str] = []
    if "Contract" in churn_rate_tables:
        top = churn_rate_tables["Contract"].iloc[0]
        low = churn_rate_tables["Contract"].iloc[-1]
        insights.append(f"- Higher churn: {top['Contract']} (~{top['churn_rate']:.1%})")
        insights.append(f"- Lower churn: {low['Contract']} (~{low['churn_rate']:.1%})")
    if "InternetService" in churn_rate_tables:
        top = churn_rate_tables["InternetService"].iloc[0]
        insights.append(f"- {top['InternetService']} has higher churn risk (~{top['churn_rate']:.1%})")
    if "PaymentMethod" in churn_rate_tables:
        top = churn_rate_tables["PaymentMethod"].iloc[0]
        insights.append(f"- Payment method {top['PaymentMethod']} shows higher churn (~{top['churn_rate']:.1%})")
    if "tenure" in df.columns:
        medians = df.groupby("Churn")["tenure"].median()
        if "Yes" in medians.index and "No" in medians.index:
            insights.append(f"- Median tenure: churn=Yes -> {medians['Yes']:.0f}, churn=No -> {medians['No']:.0f}")
    if "MonthlyCharges" in df.columns:
        medians = df.groupby("Churn")["MonthlyCharges"].median()
        if "Yes" in medians.index and "No" in medians.index:
            insights.append(f"- Median MonthlyCharges: churn=Yes -> {medians['Yes']:.2f}, churn=No -> {medians['No']:.2f}")

    overview_table = pd.DataFrame(
        [
            ["rows", shape[0]],
            ["cols", shape[1]],
            ["duplicate rows", duplicate_rows],
            ["duplicate customerID", duplicate_customer_ids],
            ["blank TotalCharges (as string)", total_charges_blank],
            ["churn rate (Yes)", f"{churn_rate:.3f}"],
        ],
        columns=["item", "value"],
    )

    lines: list[str] = []
    lines.append("# Phase 1 - EDA")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append(markdown_table(overview_table))
    lines.append("")
    lines.append("![Target](figures/target_churn.png)")
    lines.append("")
    lines.append("## Column types")
    lines.append(markdown_table(dtypes_table))
    lines.append("")
    lines.append("## Missing values")
    if len(missing_table) == 0:
        lines.append("- No explicit NaN values were found, but `TotalCharges` contains blank strings in the raw CSV.")
    else:
        lines.append(markdown_table(missing_table))
    lines.append("")
    lines.append("## Numeric features")
    lines.append(markdown_table(numeric_summary.round(3)))
    lines.append("")
    for column in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if (figures_dir / f"hist_{column}.png").exists():
            lines.append(f"![{column}](figures/hist_{column}.png)")
            lines.append("")
    lines.append("## Numeric correlation")
    lines.append("![Corr](figures/corr_heatmap.png)")
    lines.append("")
    lines.append("## Numeric vs target")
    for column in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if (figures_dir / f"box_{column}_by_churn.png").exists():
            lines.append(f"![{column}](figures/box_{column}_by_churn.png)")
            lines.append("")
    lines.append("## Churn rate by key categorical features")
    for column in key_cols:
        if column in churn_rate_tables:
            lines.append(f"### {column}")
            lines.append(f"![{column}](figures/churnrate_{column}.png)")
            lines.append("")
            lines.append(markdown_table(churn_rate_tables[column].round(4)))
            lines.append("")
    lines.append("## Quick notes")
    if insights:
        lines.extend(insights)
    else:
        lines.append("- Main outputs are saved under `reports/figures/`.")

    (reports_dir / "eda.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

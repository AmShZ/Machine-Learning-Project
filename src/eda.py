import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy()]
    return "\n".join([header, sep, *rows])


def save_hist(series: pd.Series, out: Path, title: str, bins: int = 30) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(series.dropna().to_numpy(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_bar_counts(series: pd.Series, out: Path, title: str) -> None:
    vc = series.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vc.index.astype(str), vc.to_numpy())
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_churn_rate(df: pd.DataFrame, col: str, out: Path) -> pd.DataFrame:
    tmp = df.copy()
    tmp["churn"] = (tmp["Churn"] == "Yes").astype(int)
    rates = tmp.groupby(col, dropna=False)["churn"].mean().sort_values(ascending=False)
    tbl = rates.reset_index().rename(columns={"churn": "churn_rate"}).copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tbl[col].astype(str), tbl["churn_rate"].to_numpy())
    ax.set_title(f"Churn rate by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("churn rate")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return tbl


def save_boxplot_by_target(df: pd.DataFrame, feature: str, out: Path) -> None:
    yes = df.loc[df["Churn"] == "Yes", feature].dropna().to_numpy()
    no = df.loc[df["Churn"] == "No", feature].dropna().to_numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([no, yes], labels=["No", "Yes"], showfliers=False)
    ax.set_title(f"{feature} vs Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel(feature)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def save_corr_heatmap(df_num: pd.DataFrame, out: Path) -> pd.DataFrame:
    corr = df_num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(corr.to_numpy(), aspect="auto")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Correlation heatmap (numeric)")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return corr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--reports", type=str, default="reports")
    args = parser.parse_args()

    data_path = Path(args.data)
    reports_dir = Path(args.reports)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    shape = df.shape
    dup_rows = int(df.duplicated().sum())
    dup_customer = int(df["customerID"].duplicated().sum()) if "customerID" in df.columns else 0

    dtypes_tbl = (
        df.dtypes.astype(str)
        .value_counts()
        .rename_axis("dtype")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    missing_tbl = (
        df.isna()
        .sum()
        .rename("missing")
        .to_frame()
        .query("missing > 0")
        .reset_index()
        .rename(columns={"index": "column"})
    )

    totalcharges_blank = 0
    if "TotalCharges" in df.columns:
        totalcharges_blank = int((df["TotalCharges"].astype(str).str.strip() == "").sum())
        df["TotalCharges_num"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    else:
        df["TotalCharges_num"] = np.nan

    numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges_num"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    df_num = df[numeric_cols].copy()

    numeric_summary = df_num.describe().T.reset_index().rename(columns={"index": "feature"})
    numeric_summary["missing"] = df_num.isna().sum().to_numpy()

    churn_rate = float((df["Churn"] == "Yes").mean())

    save_bar_counts(df["Churn"], figures_dir / "target_churn.png", "Target distribution (Churn)")

    for col in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if col in df.columns:
            save_hist(df[col], figures_dir / f"hist_{col}.png", f"Histogram: {col}")

    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c not in {"customerID", "Churn", "TotalCharges"}]
    for col in cat_cols:
        save_bar_counts(df[col], figures_dir / f"cat_{col}.png", f"Counts: {col}")

    save_corr_heatmap(df_num, figures_dir / "corr_heatmap.png")

    for col in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if col in df.columns:
            save_boxplot_by_target(df, col, figures_dir / f"box_{col}_by_churn.png")

    key_cols = ["Contract", "InternetService", "PaymentMethod", "OnlineSecurity", "TechSupport", "PaperlessBilling"]
    churn_rate_tables = {}
    for col in key_cols:
        if col in df.columns:
            churn_rate_tables[col] = save_churn_rate(df, col, figures_dir / f"churnrate_{col}.png")

    insights = []
    if "Contract" in churn_rate_tables:
        top = churn_rate_tables["Contract"].iloc[0]
        low = churn_rate_tables["Contract"].iloc[-1]
        insights.append(f"- higher churn: {top['Contract']} (~{top['churn_rate']:.1%})")
        insights.append(f"- lower churn: {low['Contract']} (~{low['churn_rate']:.1%})")
    if "InternetService" in churn_rate_tables:
        top = churn_rate_tables["InternetService"].iloc[0]
        insights.append(f"- {top['InternetService']} has higher churn risk (~{top['churn_rate']:.1%})")
    if "PaymentMethod" in churn_rate_tables:
        top = churn_rate_tables["PaymentMethod"].iloc[0]
        insights.append(f"- payment method {top['PaymentMethod']} shows higher churn (~{top['churn_rate']:.1%})")
    if "tenure" in df.columns:
        med = df.groupby("Churn")["tenure"].median()
        if "Yes" in med.index and "No" in med.index:
            insights.append(f"- median tenure: churn=Yes → {med['Yes']:.0f}, churn=No → {med['No']:.0f}")
    if "MonthlyCharges" in df.columns:
        med = df.groupby("Churn")["MonthlyCharges"].median()
        if "Yes" in med.index and "No" in med.index:
            insights.append(f"- median MonthlyCharges: churn=Yes → {med['Yes']:.2f}, churn=No → {med['No']:.2f}")

    overview_tbl = pd.DataFrame(
        [
            ["rows", shape[0]],
            ["cols", shape[1]],
            ["duplicate rows", dup_rows],
            ["duplicate customerID", dup_customer],
            ["blank TotalCharges (as string)", totalcharges_blank],
            ["churn rate (Yes)", f"{churn_rate:.3f}"],
        ],
        columns=["item", "value"],
    )

    lines = []
    lines.append("# Phase 1 — EDA")
    lines.append("")
    lines.append("## Dataset summary")
    lines.append(markdown_table(overview_tbl))
    lines.append("")
    lines.append("![Target](figures/target_churn.png)")
    lines.append("")
    lines.append("## Column types")
    lines.append(markdown_table(dtypes_tbl))
    lines.append("")
    lines.append("## Missing values")
    if len(missing_tbl) == 0:
        lines.append("- No explicit NaN values found, but `TotalCharges` contains blank strings in the raw CSV.")
    else:
        lines.append(markdown_table(missing_tbl))
    lines.append("")
    lines.append("## Numeric features")
    lines.append(markdown_table(numeric_summary.round(3)))
    lines.append("")
    for col in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if (figures_dir / f"hist_{col}.png").exists():
            lines.append(f"![{col}](figures/hist_{col}.png)")
            lines.append("")
    lines.append("## Numeric correlation")
    lines.append("![Corr](figures/corr_heatmap.png)")
    lines.append("")
    lines.append("## Numeric vs target")
    for col in ["tenure", "MonthlyCharges", "TotalCharges_num"]:
        if (figures_dir / f"box_{col}_by_churn.png").exists():
            lines.append(f"![{col}](figures/box_{col}_by_churn.png)")
            lines.append("")
    lines.append("## Churn rate by key categorical features")
    for col in key_cols:
        if col in churn_rate_tables:
            lines.append(f"### {col}")
            lines.append(f"![{col}](figures/churnrate_{col}.png)")
            lines.append("")
            lines.append(markdown_table(churn_rate_tables[col].round(4)))
            lines.append("")
    lines.append("## Quick notes")
    if insights:
        lines.extend(insights)
    else:
        lines.append("- Main outputs are saved under `reports/figures/`.")

    (reports_dir / "eda.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

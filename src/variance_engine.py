"""
Variance Detection Engine
Detects significant budget vs actual deviations using rule-based + statistical methods.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ── Thresholds ────────────────────────────────────────────────────────────────
RULE_THRESHOLD_PCT  = 0.10   # 10% = flag
STAT_ZSCORE_CUTOFF  = 2.0    # Z-score > 2 = statistical outlier
MIN_AMOUNT_FLAG     = 5_000  # Don't flag tiny dollar variances


SEVERITY_LEVELS = {
    "CRITICAL": {"min_pct": 0.30, "color": "#ef4444"},
    "HIGH":     {"min_pct": 0.20, "color": "#f97316"},
    "MEDIUM":   {"min_pct": 0.10, "color": "#eab308"},
    "LOW":      {"min_pct": 0.00, "color": "#22c55e"},
}


def classify_severity(abs_pct: float) -> str:
    if abs_pct >= 0.30: return "CRITICAL"
    if abs_pct >= 0.20: return "HIGH"
    if abs_pct >= 0.10: return "MEDIUM"
    return "LOW"


def load_data(path: str = "data/processed/erp_combined.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["variance_amount"] = df["actual_amount"] - df["budget_amount"]
    df["variance_pct"]    = df["variance_amount"] / df["budget_amount"].replace(0, np.nan)
    df["abs_variance_pct"]= df["variance_pct"].abs()
    return df


def rule_based_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag rows where |variance %| > threshold AND dollar amount is meaningful."""
    df = df.dropna(subset=["actual_amount"])
    flagged = df[
        (df["abs_variance_pct"] >= RULE_THRESHOLD_PCT) &
        (df["variance_amount"].abs() >= MIN_AMOUNT_FLAG)
    ].copy()
    flagged["flag_type"] = "RULE_BASED"
    return flagged


def statistical_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Z-score based outlier detection per account across all periods."""
    df = df.dropna(subset=["actual_amount"])
    results = []
    for acct_id, group in df.groupby("account_id"):
        if len(group) < 4:
            continue
        z_scores = np.abs(stats.zscore(group["variance_pct"].fillna(0)))
        outliers = group[z_scores > STAT_ZSCORE_CUTOFF].copy()
        outliers["flag_type"] = "STATISTICAL"
        outliers["z_score"]   = z_scores[z_scores > STAT_ZSCORE_CUTOFF]
        results.append(outliers)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag accounts with 3+ consecutive periods of same-direction variance."""
    df = df.dropna(subset=["actual_amount"])
    results = []
    df_sorted = df.sort_values(["account_id", "department_id", "period"])
    for (acct, dept), group in df_sorted.groupby(["account_id", "department_id"]):
        if len(group) < 3:
            continue
        variances = group["variance_pct"].fillna(0).tolist()
        periods   = group["period"].tolist()
        for i in range(2, len(variances)):
            window = variances[i-2:i+1]
            if all(v > 0.05 for v in window) or all(v < -0.05 for v in window):
                row = group[group["period"] == periods[i]].copy()
                row["flag_type"] = "TREND"
                row["trend_direction"] = "OVER" if window[-1] > 0 else "UNDER"
                results.append(row)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def build_variance_report(df: pd.DataFrame) -> pd.DataFrame:
    """Combine all flag types, deduplicate, score severity."""
    rule_flags  = rule_based_flags(df)
    stat_flags  = statistical_flags(df)
    trend_flags_ = trend_flags(df)

    all_flags = pd.concat([rule_flags, stat_flags, trend_flags_], ignore_index=True)

    # Deduplicate: keep worst flag per (account, dept, period)
    if "z_score" not in all_flags.columns:
        all_flags["z_score"] = np.nan
    if "trend_direction" not in all_flags.columns:
        all_flags["trend_direction"] = np.nan

    all_flags["severity"] = all_flags["abs_variance_pct"].apply(classify_severity)

    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    all_flags["severity_rank"] = all_flags["severity"].map(severity_order)

    # A row tripping several methods is kept once, under the first method in
    # concat order (RULE_BASED, STATISTICAL, TREND). drop_duplicates keeps the
    # first occurrence, so no sort may run before it. Then rank by severity
    # (by rank, not alphabetically, which would put LOW before MEDIUM).
    report = (
        all_flags
        .drop_duplicates(subset=["account_id", "department_id", "period"], keep="first")
        .sort_values(["severity_rank", "abs_variance_pct"], ascending=[True, False], kind="stable")
        .drop(columns=["severity_rank"])
        .reset_index(drop=True)
    )

    # Add direction label
    report["direction"] = report["variance_pct"].apply(
        lambda x: "OVER_BUDGET" if x > 0 else "UNDER_BUDGET"
    )
    # For revenue: flip direction meaning
    revenue_mask = report["account_type"] == "Revenue"
    report.loc[revenue_mask & (report["variance_pct"] > 0), "direction"] = "BEAT"
    report.loc[revenue_mask & (report["variance_pct"] < 0), "direction"] = "MISS"

    return report


def compute_summary_metrics(df: pd.DataFrame) -> dict:
    """High-level KPIs for executive dashboard."""
    df = df.dropna(subset=["actual_amount"])
    total_budget  = df["budget_amount"].sum()
    total_actual  = df["actual_amount"].sum()
    total_forecast= df["forecast_amount"].sum()

    rev_mask = df["account_type"] == "Revenue"
    rev_budget = df.loc[rev_mask, "budget_amount"].sum()
    rev_actual = df.loc[rev_mask, "actual_amount"].sum()

    exp_mask = df["account_type"].isin(["COGS", "OpEx"])
    exp_budget = df.loc[exp_mask, "budget_amount"].sum()
    exp_actual = df.loc[exp_mask, "actual_amount"].sum()

    return {
        "total_budget":         round(total_budget, 0),
        "total_actual":         round(total_actual, 0),
        "total_variance":       round(total_actual - total_budget, 0),
        "total_variance_pct":   round((total_actual - total_budget) / total_budget, 4),
        "revenue_budget":       round(rev_budget, 0),
        "revenue_actual":       round(rev_actual, 0),
        "revenue_variance_pct": round((rev_actual - rev_budget) / rev_budget, 4),
        "expense_budget":       round(exp_budget, 0),
        "expense_actual":       round(exp_actual, 0),
        "expense_variance_pct": round((exp_actual - exp_budget) / exp_budget, 4),
        "forecast_vs_budget":   round((total_forecast - total_budget) / total_budget, 4),
    }


def get_top_variances(report: pd.DataFrame, n: int = 10) -> list[dict]:
    """Return top N variances as list of dicts for LLM context."""
    top = report.head(n)
    records = []
    for _, row in top.iterrows():
        records.append({
            "account":      row["account_name"],
            "account_type": row["account_type"],
            "department":   row["department_name"],
            "period":       row["period"],
            "budget":       f"${row['budget_amount']:,.0f}",
            "actual":       f"${row['actual_amount']:,.0f}",
            "variance_amt": f"${row['variance_amount']:+,.0f}",
            "variance_pct": f"{row['variance_pct']:+.1%}",
            "severity":     row["severity"],
            "direction":    row["direction"],
            "flag_type":    row["flag_type"],
        })
    return records


if __name__ == "__main__":
    df = load_data()
    report = build_variance_report(df)
    metrics = compute_summary_metrics(df)

    print("\n── Summary Metrics ──")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n── Flagged Variances: {len(report)} ──")
    print(report[["account_name","department_name","period","variance_pct","severity","direction"]].head(15).to_string())

    report.to_csv("data/processed/variance_report.csv", index=False)
    print("\n✅ Saved to data/processed/variance_report.csv")

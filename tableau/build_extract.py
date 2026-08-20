"""
Builds the Tableau-ready .hyper extract for the Budget Variance Analyzer.

Reuses the project's own variance_engine (same statistical/rule/trend flagging
logic as the Streamlit app) so the Tableau workbook shows exactly the same
numbers as the live app, instead of re-deriving the analysis separately.

Produces two tables inside tableau/extract/budget_variance.hyper:
  - Extract  : wide grain (period x account x department) with budget/actual/
               forecast, variance calcs, flags, severity. Powers the KPI tiles,
               flagged-variance table, heatmap and severity mix.
  - Trend    : long/melted grain (period x account x department x metric_type)
               with a single `amount` measure, where metric_type in
               {Budget, Actual, Forecast}. Actual rows exist only for closed
               periods and Forecast rows only for open periods, so a Line mark
               naturally breaks at the actual/forecast boundary without any
               "connect null values" trickery. Powers the trend chart and the
               budget-vs-actual-by-type bar chart.

Usage:
    python tableau/build_extract.py
"""
import os
import sys
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from generate_data import CURRENT_PERIOD  # noqa: E402
from variance_engine import load_data, build_variance_report, classify_severity  # noqa: E402

from tableauhyperapi import (
    HyperProcess, Telemetry, Connection, CreateMode,
    TableDefinition, TableName, SqlType, Inserter, NOT_NULLABLE, NULLABLE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HYPER_PATH = os.path.join(HERE, "extract", "budget_variance.hyper")


def period_to_date(period: str) -> datetime.date:
    y, m = period.split("-")
    return datetime.date(int(y), int(m), 1)


def build_wide_table() -> pd.DataFrame:
    df = load_data(os.path.join(ROOT, "data", "processed", "erp_combined.csv"))
    df["account_id"] = df["account_id"].astype(str)
    report = build_variance_report(df)
    report["account_id"] = report["account_id"].astype(str)

    flag_cols = ["account_id", "department_id", "period", "flag_type", "severity", "z_score", "trend_direction"]
    flags = report[flag_cols].rename(columns={"severity": "flag_severity"})

    df = df.merge(flags, on=["account_id", "department_id", "period"], how="left")
    df["is_flagged"] = df["flag_type"].notna()

    # Row-level severity for every closed-period row (not just flagged ones) —
    # useful for tooltips/coloring even where a row didn't cross the flag threshold.
    has_actual = df["actual_amount"].notna()
    df["severity"] = np.where(has_actual, df["abs_variance_pct"].apply(
        lambda v: classify_severity(v) if pd.notna(v) else None), None)

    # Direction, with revenue beat/miss flip (mirrors variance_engine.build_variance_report)
    direction = np.where(df["variance_pct"] > 0, "OVER_BUDGET", "UNDER_BUDGET")
    revenue_mask = df["account_type"] == "Revenue"
    direction = np.where(revenue_mask & (df["variance_pct"] > 0), "BEAT", direction)
    direction = np.where(revenue_mask & (df["variance_pct"] < 0), "MISS", direction)
    df["direction"] = np.where(has_actual, direction, None)

    df["is_closed_period"] = df["period"] <= CURRENT_PERIOD
    df["period_date"] = df["period"].apply(period_to_date)
    df["month_num"] = df["period"].str.slice(5, 7).astype(int)

    cols = [
        "period", "period_date", "month_num",
        "account_id", "account_name", "account_type",
        "department_id", "department_name",
        "budget_amount", "actual_amount", "forecast_amount",
        "variance_amount", "variance_pct", "abs_variance_pct",
        "direction", "severity",
        "is_flagged", "flag_type", "z_score", "trend_direction",
        "is_closed_period",
    ]
    return df[cols].sort_values(["period", "account_id", "department_id"]).reset_index(drop=True)


def build_trend_table(wide: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["period", "period_date", "month_num", "account_id", "account_name",
                 "department_id", "department_name", "is_closed_period"]

    budget = wide[base_cols + ["budget_amount"]].rename(columns={"budget_amount": "amount"}).copy()
    budget["metric_type"] = "Budget"

    actual = wide.loc[wide["is_closed_period"], base_cols + ["actual_amount"]] \
        .rename(columns={"actual_amount": "amount"}).copy()
    actual["metric_type"] = "Actual"

    forecast = wide.loc[~wide["is_closed_period"], base_cols + ["forecast_amount"]] \
        .rename(columns={"forecast_amount": "amount"}).copy()
    forecast["metric_type"] = "Forecast"

    trend = pd.concat([budget, actual, forecast], ignore_index=True)
    trend = trend.dropna(subset=["amount"])
    return trend[base_cols + ["metric_type", "amount"]].sort_values(
        ["account_id", "department_id", "period", "metric_type"]).reset_index(drop=True)


def write_hyper(wide: pd.DataFrame, trend: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)

    extract_def = TableDefinition(
        table_name=TableName("Extract"),
        columns=[
            TableDefinition.Column("period", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("period_date", SqlType.date(), NOT_NULLABLE),
            TableDefinition.Column("month_num", SqlType.int(), NOT_NULLABLE),
            TableDefinition.Column("account_id", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("account_name", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("account_type", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("department_id", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("department_name", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("budget_amount", SqlType.double(), NOT_NULLABLE),
            TableDefinition.Column("actual_amount", SqlType.double(), NULLABLE),
            TableDefinition.Column("forecast_amount", SqlType.double(), NOT_NULLABLE),
            TableDefinition.Column("variance_amount", SqlType.double(), NULLABLE),
            TableDefinition.Column("variance_pct", SqlType.double(), NULLABLE),
            TableDefinition.Column("abs_variance_pct", SqlType.double(), NULLABLE),
            TableDefinition.Column("direction", SqlType.text(), NULLABLE),
            TableDefinition.Column("severity", SqlType.text(), NULLABLE),
            TableDefinition.Column("is_flagged", SqlType.bool(), NOT_NULLABLE),
            TableDefinition.Column("flag_type", SqlType.text(), NULLABLE),
            TableDefinition.Column("z_score", SqlType.double(), NULLABLE),
            TableDefinition.Column("trend_direction", SqlType.text(), NULLABLE),
            TableDefinition.Column("is_closed_period", SqlType.bool(), NOT_NULLABLE),
        ],
    )

    trend_def = TableDefinition(
        table_name=TableName("Trend"),
        columns=[
            TableDefinition.Column("period", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("period_date", SqlType.date(), NOT_NULLABLE),
            TableDefinition.Column("month_num", SqlType.int(), NOT_NULLABLE),
            TableDefinition.Column("account_id", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("account_name", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("department_id", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("department_name", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("is_closed_period", SqlType.bool(), NOT_NULLABLE),
            TableDefinition.Column("metric_type", SqlType.text(), NOT_NULLABLE),
            TableDefinition.Column("amount", SqlType.double(), NOT_NULLABLE),
        ],
    )

    def row_of(rec, columns):
        out = []
        for c in columns:
            v = rec[c]
            if isinstance(v, float) and np.isnan(v):
                out.append(None)
            elif pd.isna(v) if not isinstance(v, (list, dict)) else False:
                out.append(None)
            else:
                out.append(v)
        return out

    with HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU) as hp:
        with Connection(endpoint=hp.endpoint, database=path, create_mode=CreateMode.CREATE_AND_REPLACE) as conn:
            conn.catalog.create_table(extract_def)
            conn.catalog.create_table(trend_def)

            ext_cols = [c.name.unescaped for c in extract_def.columns]
            with Inserter(conn, extract_def) as inserter:
                for _, rec in wide.iterrows():
                    inserter.add_row(row_of(rec, ext_cols))
                inserter.execute()

            trend_cols = [c.name.unescaped for c in trend_def.columns]
            with Inserter(conn, trend_def) as inserter:
                for _, rec in trend.iterrows():
                    inserter.add_row(row_of(rec, trend_cols))
                inserter.execute()

    print(f"Wrote {path}")
    print(f"  Extract rows: {len(wide)}")
    print(f"  Trend rows:   {len(trend)}")


def main():
    wide = build_wide_table()
    trend = build_trend_table(wide)
    write_hyper(wide, trend, HYPER_PATH)


if __name__ == "__main__":
    main()

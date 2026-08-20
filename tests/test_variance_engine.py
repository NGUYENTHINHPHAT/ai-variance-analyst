"""
Regression tests for the deterministic core of the project: variance_engine.py.

These are plain unit tests against synthetic, hand-computed fixtures (no API
calls, no real data file needed) so they run fast and stay exact. Thresholds
mirror the constants in variance_engine.py (RULE_THRESHOLD_PCT=10%,
STAT_ZSCORE_CUTOFF=2.0, MIN_AMOUNT_FLAG=$5,000) by name, not by hardcoded
duplicate numbers, so a deliberate threshold change here fails loudly instead
of silently drifting from production behavior.
"""
import numpy as np
import pandas as pd
import pytest

from variance_engine import (
    classify_severity,
    load_data,
    rule_based_flags,
    statistical_flags,
    trend_flags,
    build_variance_report,
    compute_summary_metrics,
    get_top_variances,
    RULE_THRESHOLD_PCT,
    STAT_ZSCORE_CUTOFF,
    MIN_AMOUNT_FLAG,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_row(period="2024-01", account_id="4001", account_name="Test Account",
             account_type="OpEx", department_id="D01", department_name="Test Dept",
             budget_amount=100_000.0, actual_amount=None, variance_amount=0.0,
             variance_pct=0.0, abs_variance_pct=0.0):
    if actual_amount is None:
        actual_amount = budget_amount + variance_amount
    return dict(
        period=period, account_id=account_id, account_name=account_name,
        account_type=account_type, department_id=department_id,
        department_name=department_name, budget_amount=budget_amount,
        actual_amount=actual_amount, forecast_amount=actual_amount,
        variance_amount=variance_amount, variance_pct=variance_pct,
        abs_variance_pct=abs_variance_pct,
    )


def make_df(rows):
    return pd.DataFrame(rows)


# ── classify_severity ────────────────────────────────────────────────────────

@pytest.mark.parametrize("pct, expected", [
    (0.0, "LOW"),
    (0.0999, "LOW"),
    (0.10, "MEDIUM"),       # lower bound is inclusive
    (0.1999, "MEDIUM"),
    (0.20, "HIGH"),         # lower bound is inclusive
    (0.2999, "HIGH"),
    (0.30, "CRITICAL"),     # lower bound is inclusive
    (1.5, "CRITICAL"),
])
def test_classify_severity_boundaries(pct, expected):
    assert classify_severity(pct) == expected


# ── load_data ────────────────────────────────────────────────────────────────

def test_load_data_computes_variance_columns(tmp_path):
    csv_path = tmp_path / "erp.csv"
    pd.DataFrame([
        {"period": "2024-01", "account_id": "4001", "account_name": "Rev",
         "account_type": "Revenue", "department_id": "D01", "department_name": "Sales",
         "budget_amount": 100_000.0, "actual_amount": 115_000.0, "forecast_amount": 115_000.0},
        # zero budget must not raise ZeroDivisionError -> should become NaN, not inf
        {"period": "2024-01", "account_id": "4002", "account_name": "ZeroBudget",
         "account_type": "OpEx", "department_id": "D01", "department_name": "Sales",
         "budget_amount": 0.0, "actual_amount": 500.0, "forecast_amount": 500.0},
    ]).to_csv(csv_path, index=False)

    df = load_data(str(csv_path))

    row = df.iloc[0]
    assert row["variance_amount"] == pytest.approx(15_000.0)
    assert row["variance_pct"] == pytest.approx(0.15)
    assert row["abs_variance_pct"] == pytest.approx(0.15)

    zero_row = df.iloc[1]
    assert pd.isna(zero_row["variance_pct"])
    assert pd.isna(zero_row["abs_variance_pct"])


# ── rule_based_flags ─────────────────────────────────────────────────────────

def test_rule_based_flags_requires_both_pct_and_dollar_threshold():
    df = make_df([
        # exactly at both thresholds -> flags (inclusive >=)
        make_row(account_id="A", variance_amount=MIN_AMOUNT_FLAG,
                  variance_pct=RULE_THRESHOLD_PCT, abs_variance_pct=RULE_THRESHOLD_PCT),
        # big % but trivial dollar amount -> should NOT flag
        make_row(account_id="B", budget_amount=1_000.0, variance_amount=200.0,
                  variance_pct=0.20, abs_variance_pct=0.20),
        # big dollar amount but small % -> should NOT flag
        make_row(account_id="C", budget_amount=1_000_000.0, variance_amount=50_000.0,
                  variance_pct=0.05, abs_variance_pct=0.05),
        # open/forecast period with no actual yet -> dropped, not flagged, no crash
        make_row(account_id="D", actual_amount=np.nan, variance_amount=np.nan,
                  variance_pct=np.nan, abs_variance_pct=np.nan),
    ])

    flagged = rule_based_flags(df)

    assert set(flagged["account_id"]) == {"A"}
    assert (flagged["flag_type"] == "RULE_BASED").all()


def test_rule_based_flags_empty_when_nothing_qualifies():
    df = make_df([
        make_row(account_id="A", variance_amount=1_000.0, variance_pct=0.03, abs_variance_pct=0.03),
    ])
    assert rule_based_flags(df).empty


# ── statistical_flags ────────────────────────────────────────────────────────

def test_statistical_flags_detects_outlier_within_account_history():
    # One account with 6 periods: five clustered near 0, one clear outlier.
    # z-scores hand-verified with scipy.stats.zscore: outlier ~2.24, rest ~0.4-0.5.
    variance_pcts = [0.01, -0.02, 0.015, -0.01, 0.02, 2.0]
    rows = [
        make_row(account_id="OUTLIER_ACCT", period=f"2024-{i+1:02d}",
                  variance_pct=v, abs_variance_pct=abs(v),
                  variance_amount=v * 100_000)
        for i, v in enumerate(variance_pcts)
    ]
    # A second account with too few periods (<4) must be skipped entirely, not error.
    rows += [
        make_row(account_id="TOO_SHORT", period=f"2024-{i+1:02d}", variance_pct=5.0,
                  abs_variance_pct=5.0, variance_amount=500_000)
        for i in range(3)
    ]
    df = make_df(rows)

    flagged = statistical_flags(df)

    assert set(flagged["account_id"]) == {"OUTLIER_ACCT"}
    assert len(flagged) == 1
    assert flagged.iloc[0]["period"] == "2024-06"
    assert flagged.iloc[0]["flag_type"] == "STATISTICAL"
    assert flagged.iloc[0]["z_score"] > STAT_ZSCORE_CUTOFF


def test_statistical_flags_returns_empty_df_when_no_group_qualifies():
    df = make_df([
        make_row(account_id="A", period=f"2024-{i+1:02d}") for i in range(2)
    ])
    result = statistical_flags(df)
    assert result.empty


# ── trend_flags ──────────────────────────────────────────────────────────────

def test_trend_flags_fires_on_three_consecutive_same_direction_periods():
    # variance_pct > 0.05 for 4 straight months -> flags periods 3 and 4
    # (the two windows [1,2,3] and [2,3,4] are both all-positive).
    rows = [
        make_row(account_id="A", department_id="D1", period=p, variance_pct=v, abs_variance_pct=abs(v))
        for p, v in zip(["2024-01", "2024-02", "2024-03", "2024-04"], [0.06, 0.07, 0.08, 0.09])
    ]
    df = make_df(rows)

    flagged = trend_flags(df)

    assert list(flagged["period"]) == ["2024-03", "2024-04"]
    assert (flagged["flag_type"] == "TREND").all()
    assert (flagged["trend_direction"] == "OVER").all()


def test_trend_flags_ignores_alternating_direction():
    rows = [
        make_row(account_id="A", department_id="D1", period=p, variance_pct=v, abs_variance_pct=abs(v))
        for p, v in zip(["2024-01", "2024-02", "2024-03", "2024-04"], [0.06, -0.06, 0.06, -0.06])
    ]
    df = make_df(rows)
    assert trend_flags(df).empty


def test_trend_flags_direction_under():
    rows = [
        make_row(account_id="A", department_id="D1", period=p, variance_pct=v, abs_variance_pct=abs(v))
        for p, v in zip(["2024-01", "2024-02", "2024-03"], [-0.06, -0.07, -0.08])
    ]
    df = make_df(rows)
    flagged = trend_flags(df)
    assert len(flagged) == 1
    assert flagged.iloc[0]["trend_direction"] == "UNDER"


# ── build_variance_report ────────────────────────────────────────────────────

def test_build_variance_report_dedupes_row_flagged_by_multiple_methods():
    # Same (account, dept, period) crosses both the rule-based threshold AND
    # is a statistical outlier in its account's history -> must appear ONCE.
    variance_pcts = [0.01, -0.02, 0.015, -0.01, 0.02, 0.50]  # last one is both rule + stat flagged
    rows = [
        make_row(account_id="DUP", department_id="D1", period=f"2024-{i+1:02d}",
                  variance_pct=v, abs_variance_pct=abs(v), variance_amount=v * 100_000)
        for i, v in enumerate(variance_pcts)
    ]
    df = make_df(rows)

    report = build_variance_report(df)
    dup_rows = report[(report["account_id"] == "DUP") & (report["period"] == "2024-06")]

    assert len(dup_rows) == 1
    assert dup_rows.iloc[0]["severity"] == classify_severity(0.50)


@pytest.mark.parametrize("account_type, variance_pct, expected_direction", [
    ("Revenue", 0.15, "BEAT"),
    ("Revenue", -0.15, "MISS"),
    ("OpEx", 0.15, "OVER_BUDGET"),
    ("OpEx", -0.15, "UNDER_BUDGET"),
    ("COGS", 0.15, "OVER_BUDGET"),
])
def test_build_variance_report_direction_labeling(account_type, variance_pct, expected_direction):
    df = make_df([
        make_row(account_id="A", account_type=account_type, variance_pct=variance_pct,
                  abs_variance_pct=abs(variance_pct), variance_amount=variance_pct * 100_000),
    ])
    report = build_variance_report(df)
    assert report.iloc[0]["direction"] == expected_direction


def test_build_variance_report_empty_input_does_not_crash():
    # No flaggable rows at all (but a real schema, as the pipeline always provides)
    # -> should return an empty report, not raise.
    df = make_df([
        make_row(account_id="A", variance_pct=0.01, abs_variance_pct=0.01, variance_amount=1_000.0),
    ])
    report = build_variance_report(df)
    assert report.empty


# ── compute_summary_metrics ──────────────────────────────────────────────────

def test_compute_summary_metrics_totals_and_splits():
    df = make_df([
        make_row(account_id="REV1", account_type="Revenue", budget_amount=200_000.0,
                  actual_amount=220_000.0),
        make_row(account_id="OPEX1", account_type="OpEx", budget_amount=50_000.0,
                  actual_amount=55_000.0),
        make_row(account_id="COGS1", account_type="COGS", budget_amount=30_000.0,
                  actual_amount=33_000.0),
        # open period with no actual yet -> must be excluded from every total
        make_row(account_id="OPEN1", account_type="OpEx", budget_amount=999_999.0,
                  actual_amount=np.nan),
    ])

    metrics = compute_summary_metrics(df)

    assert metrics["total_budget"] == pytest.approx(280_000.0)
    assert metrics["total_actual"] == pytest.approx(308_000.0)
    assert metrics["total_variance"] == pytest.approx(28_000.0)
    assert metrics["revenue_budget"] == pytest.approx(200_000.0)
    assert metrics["revenue_actual"] == pytest.approx(220_000.0)
    assert metrics["revenue_variance_pct"] == pytest.approx(0.10)
    assert metrics["expense_budget"] == pytest.approx(80_000.0)
    assert metrics["expense_actual"] == pytest.approx(88_000.0)
    assert metrics["expense_variance_pct"] == pytest.approx(0.10)


# ── get_top_variances ────────────────────────────────────────────────────────

def test_get_top_variances_formats_and_respects_n():
    report = make_df([
        make_row(account_id="A", account_name="Marketing Spend", department_name="Marketing",
                  budget_amount=100_000.0, actual_amount=115_000.0, variance_amount=15_000.0,
                  variance_pct=0.15, abs_variance_pct=0.15),
        make_row(account_id="B", account_name="Cloud Hosting", department_name="Engineering",
                  budget_amount=50_000.0, actual_amount=40_000.0, variance_amount=-10_000.0,
                  variance_pct=-0.20, abs_variance_pct=0.20),
        make_row(account_id="C", account_name="Should Be Excluded", department_name="Ops",
                  budget_amount=1.0, actual_amount=1.0),
    ])
    report["severity"] = report["abs_variance_pct"].apply(classify_severity)
    report["direction"] = "OVER_BUDGET"
    report["flag_type"] = "RULE_BASED"

    top = get_top_variances(report, n=2)

    assert len(top) == 2
    assert top[0]["account"] == "Marketing Spend"
    assert top[0]["budget"] == "$100,000"
    assert top[0]["actual"] == "$115,000"
    assert top[0]["variance_amt"] == "$+15,000"
    assert top[0]["variance_pct"] == "+15.0%"
    assert top[1]["variance_amt"] == "$-10,000"
    assert top[1]["variance_pct"] == "-20.0%"
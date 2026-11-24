"""
Tests for the DuckDB SQL layer (sql/ + src/sql_engine.py + src/seed_erp_db.py).

Most tests build a tiny in-memory ERP so each rule is checked against numbers
you can verify by hand. The parity test at the bottom seeds a real database from
the committed data/raw CSVs and checks the SQL results match the Python engine.
"""
import logging
import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from seed_erp_db import build_database, create_schema
from sql_engine import (
    DATA_QUALITY_CHECKS, SQL_DIR, build_variance_report_sql,
    load_variance_base, run_data_quality_checks,
)
from variance_engine import build_variance_report, load_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCKED, OPEN = "2024-01", "2024-02"


@pytest.fixture
def con():
    """Two accounts, two departments, January locked, February open, no entries yet."""
    con = duckdb.connect(":memory:")
    create_schema(con)
    con.execute("""
        INSERT INTO coa_mapping VALUES
            ('4001', 'Product Revenue - SaaS', 'Revenue', 'credit'),
            ('6003', 'Marketing & Advertising', 'OpEx', 'debit');
        INSERT INTO departments VALUES ('D02', 'Sales'), ('D03', 'Marketing');
        INSERT INTO close_control_log VALUES
            ('2024-01', 'LOCKED', TIMESTAMP '2024-02-06 18:00:00', 'controller'),
            ('2024-02', 'OPEN', NULL, NULL);
        INSERT INTO budget_targets VALUES
            ('2024-01', '4001', 'D02', 1000.00),
            ('2024-01', '6003', 'D03', 500.00),
            ('2024-02', '4001', 'D02', 1000.00);
        INSERT INTO forecast_targets VALUES
            ('2024-01', '4001', 'D02', 1000.00),
            ('2024-01', '6003', 'D03', 500.00),
            ('2024-02', '4001', 'D02', 950.00);
    """)
    yield con
    con.close()


def add_entry(con, entry_id, period, gl_code, dept, amount, status="POSTED"):
    con.execute("INSERT INTO gl_entries VALUES (?, ?, ?, ?, ?, ?, NULL)",
                [entry_id, period, gl_code, dept, amount, status])


def row(base, period, account_id, dept):
    match = base[(base["period"] == period) & (base["account_id"] == account_id)
                 & (base["department_id"] == dept)]
    assert len(match) == 1, f"expected one row for {period}/{account_id}/{dept}"
    return match.iloc[0]


def check_result(con, name):
    return next(r for r in run_data_quality_checks(con) if r.check.name == name)


# ── Extract: what counts as an actual ───────────────────────────────────────

def test_draft_entries_are_excluded_from_actuals(con):
    add_entry(con, "JE-1", LOCKED, "4001", "D02", 700.00)
    add_entry(con, "JE-2", LOCKED, "4001", "D02", 400.00)
    add_entry(con, "JE-3", LOCKED, "4001", "D02", 9_999.00, status="DRAFT")

    r = row(load_variance_base(con), LOCKED, "4001", "D02")
    assert r["actual_amount"] == pytest.approx(1_100.00)
    assert r["variance_amount"] == pytest.approx(100.00)
    assert r["variance_pct"] == pytest.approx(0.10)


def test_open_period_has_no_actuals_even_with_postings(con):
    add_entry(con, "JE-1", OPEN, "4001", "D02", 600.00)

    r = row(load_variance_base(con), OPEN, "4001", "D02")
    assert pd.isna(r["actual_amount"])
    assert pd.isna(r["variance_pct"])
    assert r["forecast_amount"] == pytest.approx(950.00)


def test_locked_period_with_nothing_posted_reports_zero_actual(con):
    r = row(load_variance_base(con), LOCKED, "6003", "D03")
    assert r["actual_amount"] == 0
    assert r["variance_amount"] == pytest.approx(-500.00)
    assert r["variance_pct"] == pytest.approx(-1.0)


def test_actuals_without_budget_get_zero_budget_and_null_pct(con):
    add_entry(con, "JE-1", LOCKED, "6003", "D02", 250.00)  # no budget line for 6003/D02

    r = row(load_variance_base(con), LOCKED, "6003", "D02")
    assert r["budget_amount"] == 0
    assert r["actual_amount"] == pytest.approx(250.00)
    assert pd.isna(r["variance_pct"])  # NULLIF guard: no divide-by-zero

    result = check_result(con, "actuals_without_budget")
    assert result.issue_count == 1
    assert result.rows.iloc[0]["gl_code"] == "6003"


# ── Data-quality checks ─────────────────────────────────────────────────────

def test_unmapped_gl_code_is_excluded_and_triggers_warning(con, caplog):
    add_entry(con, "JE-1", LOCKED, "4001", "D02", 1_000.00)
    add_entry(con, "JE-2", LOCKED, "9999", "D02", 3_000.00)  # not in coa_mapping

    base = load_variance_base(con)
    assert "9999" not in set(base["account_id"])

    with caplog.at_level(logging.WARNING, logger="sql_engine"):
        result = check_result(con, "unmapped_gl_codes")
    assert result.check.severity == "ERROR"
    assert result.issue_count == 1
    assert result.rows.iloc[0]["gl_code"] == "9999"
    assert float(result.rows.iloc[0]["total_amount"]) == pytest.approx(3_000.00)
    assert "unmapped_gl_codes" in caplog.text


def test_unmapped_gl_code_in_open_period_is_not_flagged(con):
    add_entry(con, "JE-1", OPEN, "9999", "D02", 3_000.00)
    assert check_result(con, "unmapped_gl_codes").passed


def test_draft_left_in_locked_period_is_flagged(con):
    add_entry(con, "JE-1", LOCKED, "4001", "D02", 120.00, status="DRAFT")
    add_entry(con, "JE-2", OPEN, "4001", "D02", 80.00, status="DRAFT")  # fine: period still open

    result = check_result(con, "draft_entries_in_locked_periods")
    assert result.issue_count == 1
    assert result.rows.iloc[0]["entry_id"] == "JE-1"


def test_unknown_department_and_duplicate_postings_are_flagged(con):
    add_entry(con, "JE-1", LOCKED, "4001", "D99", 10.00)
    add_entry(con, "JE-2", LOCKED, "4001", "D02", 450.00)
    add_entry(con, "JE-3", LOCKED, "4001", "D02", 450.00)

    assert check_result(con, "unknown_departments").rows.iloc[0]["department_id"] == "D99"
    dupes = check_result(con, "possible_duplicate_entries")
    assert dupes.issue_count == 1
    assert dupes.rows.iloc[0]["entry_ids"] == "JE-2, JE-3"


def test_clean_ledger_passes_every_check(con):
    add_entry(con, "JE-1", LOCKED, "4001", "D02", 1_000.00)
    add_entry(con, "JE-2", LOCKED, "6003", "D03", 480.00)
    assert all(r.passed for r in run_data_quality_checks(con))


def test_every_registered_check_has_a_sql_file():
    for check in DATA_QUALITY_CHECKS:
        assert os.path.exists(os.path.join(SQL_DIR, "checks", f"{check.name}.sql")), check.name


# ── Parity with the Python engine on the real generated data ────────────────

@pytest.fixture(scope="module")
def seeded_db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("erp") / "erp.duckdb")
    build_database(db_path=path, raw_dir=os.path.join(ROOT, "data", "raw"))
    con = duckdb.connect(path, read_only=True)
    yield con
    con.close()


@pytest.fixture(scope="module")
def csv_data():
    df = load_data(os.path.join(ROOT, "data", "processed", "erp_combined.csv"))
    df["account_id"] = df["account_id"].astype(str)
    return df


def test_sql_extract_matches_csv_pipeline(seeded_db, csv_data):
    """Split postings, drafts and open-period activity must net back to the generator's numbers."""
    base = load_variance_base(seeded_db)
    assert list(base.columns) == list(csv_data.columns)
    assert len(base) == len(csv_data)

    keys = ["period", "account_id", "department_id"]
    merged = base.merge(csv_data, on=keys, suffixes=("_sql", "_csv"), validate="one_to_one")
    assert len(merged) == len(csv_data)
    for col in ["budget_amount", "actual_amount", "forecast_amount", "variance_pct"]:
        np.testing.assert_allclose(merged[f"{col}_sql"], merged[f"{col}_csv"],
                                   rtol=1e-9, atol=1e-9, err_msg=col)


def test_sql_flags_match_python_engine(seeded_db, csv_data):
    sql_report = build_variance_report_sql(seeded_db)
    py_report = build_variance_report(csv_data)

    assert len(sql_report) == len(py_report) > 0
    assert list(sql_report.columns) == list(py_report.columns)

    exact = ["period", "account_id", "department_id", "flag_type",
             "trend_direction", "severity", "direction"]
    pd.testing.assert_frame_equal(
        sql_report[exact].reset_index(drop=True).fillna("-"),
        py_report[exact].reset_index(drop=True).fillna("-"),
        check_dtype=False,
    )
    for col in ["variance_amount", "variance_pct", "z_score"]:
        np.testing.assert_allclose(sql_report[col].astype(float), py_report[col].astype(float),
                                   rtol=1e-9, atol=1e-9, err_msg=col)


def test_seeded_ledger_contains_rows_the_extract_must_ignore(seeded_db):
    """Guards the parity test: it only means something if there are drafts and open-period postings."""
    drafts, open_posted = seeded_db.execute("""
        SELECT
            COUNT(*) FILTER (WHERE e.entry_status = 'DRAFT'),
            COUNT(*) FILTER (WHERE e.entry_status = 'POSTED' AND c.close_status = 'OPEN')
        FROM gl_entries e JOIN close_control_log c USING (period)
    """).fetchone()
    assert drafts > 0 and open_posted > 0

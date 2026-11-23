"""
SQL layer: runs the queries in sql/ against the mock ERP DuckDB database.

    load_variance_base(con)        -> the budget/actual/forecast table the app uses
    build_variance_report_sql(con) -> variance flags computed in SQL
    run_data_quality_checks(con)   -> pre-variance data-quality checks

The variance flags are also computed in Python (variance_engine.py);
tests/test_sql_layer.py checks that both produce the same report.
"""
import logging
import os
from dataclasses import dataclass

import duckdb
import pandas as pd

from variance_engine import MIN_AMOUNT_FLAG, RULE_THRESHOLD_PCT, STAT_ZSCORE_CUTOFF

logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQL_DIR = os.path.join(ROOT, "sql")
DEFAULT_DB_PATH = os.path.join(ROOT, "data", "erp.duckdb")

# Detection parameters not exposed as constants in variance_engine.py
TREND_THRESHOLD_PCT = 0.05
MIN_STAT_ROWS = 4


@dataclass
class DataQualityCheck:
    name: str          # also the file name under sql/checks/
    severity: str      # ERROR: numbers are wrong or incomplete; WARNING: worth a look
    description: str


DATA_QUALITY_CHECKS = [
    DataQualityCheck("unmapped_gl_codes", "ERROR",
                     "Posted entries with a GL code missing from the chart of accounts "
                     "(excluded from the P&L)"),
    DataQualityCheck("unknown_departments", "ERROR",
                     "Posted entries booked to a department that doesn't exist"),
    DataQualityCheck("draft_entries_in_locked_periods", "WARNING",
                     "Draft entries left in closed periods (excluded from actuals)"),
    DataQualityCheck("actuals_without_budget", "WARNING",
                     "Actuals with no budget line (variance % can't be computed)"),
    DataQualityCheck("possible_duplicate_entries", "WARNING",
                     "Identical posted lines that may have been entered twice"),
]


@dataclass
class DataQualityResult:
    check: DataQualityCheck
    rows: pd.DataFrame

    @property
    def passed(self) -> bool:
        return self.rows.empty

    @property
    def issue_count(self) -> int:
        return len(self.rows)


def read_sql(relative_path: str) -> str:
    with open(os.path.join(SQL_DIR, relative_path)) as f:
        return f.read()


def connect(db_path: str = DEFAULT_DB_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"{db_path} not found. Run `python src/seed_erp_db.py` first.")
    return duckdb.connect(db_path, read_only=read_only)


def load_variance_base(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Budget vs. actual vs. forecast at period x account x department grain."""
    return con.execute(read_sql("extract_variance_base.sql")).df()


def build_variance_report_sql(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Flag significant variances in SQL (same logic as variance_engine.build_variance_report)."""
    extract = read_sql("extract_variance_base.sql").strip().rstrip(";")
    con.execute(f"CREATE OR REPLACE TEMP TABLE variance_base AS {extract}")
    return con.execute(read_sql("variance_flags.sql"), {
        "rule_threshold": RULE_THRESHOLD_PCT,
        "min_amount": MIN_AMOUNT_FLAG,
        "z_cutoff": STAT_ZSCORE_CUTOFF,
        "min_stat_rows": MIN_STAT_ROWS,
        "trend_threshold": TREND_THRESHOLD_PCT,
    }).df()


def run_data_quality_checks(con: duckdb.DuckDBPyConnection) -> list:
    """Run every check in sql/checks/ and log a warning for each one that finds issues."""
    results = []
    for check in DATA_QUALITY_CHECKS:
        rows = con.execute(read_sql(os.path.join("checks", f"{check.name}.sql"))).df()
        result = DataQualityResult(check, rows)
        if not result.passed:
            logger.warning("Data quality %s: %s -- %d issue(s)",
                           check.severity, check.name, result.issue_count)
        results.append(result)
    return results

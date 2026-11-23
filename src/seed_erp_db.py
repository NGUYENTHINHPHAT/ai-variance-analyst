"""
Seeds the mock ERP DuckDB database (data/erp.duckdb) from the generator's CSVs.

generate_data.py produces one actual amount per period x account x department.
A real GL doesn't look like that, so this script reshapes it into journal-entry
lines the SQL layer has to aggregate and filter:

  * each actual is split into 2-3 POSTED lines that sum back to it to the cent
  * a few DRAFT lines are left in locked periods (abandoned accrual drafts)
  * the open period (December) gets partial-month POSTED and DRAFT activity

None of the extra rows should reach the variance analysis: drafts are filtered
by entry_status, and December by close_control_log. tests/test_sql_layer.py
checks that the SQL extract still reproduces the generator's numbers exactly.

Usage:
    python src/seed_erp_db.py
"""
import json
import os
import sys
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_data import CURRENT_PERIOD, DEPARTMENTS, PERIODS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
DEFAULT_DB_PATH = os.path.join(ROOT, "data", "erp.duckdb")
SCHEMA_PATH = os.path.join(ROOT, "sql", "schema.sql")

# Separate RNG from generate_data.py so seeding never changes the generated numbers.
SEED = 7
N_STALE_DRAFTS = 6          # drafts left behind in locked periods
OPEN_PERIOD_ACTIVITY = 0.4  # share of open-period budget lines with MTD postings


def create_schema(con: duckdb.DuckDBPyConnection) -> None:
    with open(SCHEMA_PATH) as f:
        con.execute(f.read())


def _split_cents(total_cents: int, n_parts: int, rng: np.random.Generator) -> list:
    """Split an amount into n parts that sum back to it exactly (in cents)."""
    weights = rng.dirichlet(np.ones(n_parts))
    parts = [int(total_cents * w) for w in weights[:-1]]
    parts.append(total_cents - sum(parts))
    return parts


def build_gl_entries(actuals: pd.DataFrame, budget: pd.DataFrame,
                     rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    counters = {}

    def add(period, gl_code, dept, cents, status, description):
        counters[period] = counters.get(period, 0) + 1
        rows.append({
            "entry_id": f"JE-{period.replace('-', '')}-{counters[period]:05d}",
            "period": period,
            "gl_code": gl_code,
            "department_id": dept,
            "amount": cents / 100,
            "entry_status": status,
            "description": description,
        })

    # Posted lines for every closed-period actual
    for r in actuals.itertuples(index=False):
        total_cents = int(round(r.actual_amount * 100))
        for cents in _split_cents(total_cents, int(rng.integers(2, 4)), rng):
            add(r.period, r.account_id, r.department_id, cents, "POSTED", "Monthly activity")

    # Stale drafts in locked periods: never approved, must not count as actuals
    for idx in rng.choice(len(actuals), size=N_STALE_DRAFTS, replace=False):
        r = actuals.iloc[idx]
        cents = int(round(r.actual_amount * rng.uniform(0.02, 0.10) * 100))
        add(r.period, r.account_id, r.department_id, cents, "DRAFT", "Accrual draft - not approved")

    # Open period: partial month-to-date postings plus drafts, excluded by the close filter
    open_budget = budget[budget["period"] > CURRENT_PERIOD]
    for r in open_budget.itertuples(index=False):
        if rng.random() < OPEN_PERIOD_ACTIVITY:
            cents = int(round(r.budget_amount * rng.uniform(0.3, 0.5) * 100))
            add(r.period, r.account_id, r.department_id, cents, "POSTED", "Month-to-date activity")
        if rng.random() < 0.1:
            cents = int(round(r.budget_amount * rng.uniform(0.05, 0.15) * 100))
            add(r.period, r.account_id, r.department_id, cents, "DRAFT", "Pending accrual")

    return pd.DataFrame(rows)


def build_close_log() -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        if period <= CURRENT_PERIOD:
            year, month = map(int, period.split("-"))
            next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
            rows.append({"period": period, "close_status": "LOCKED",
                         "locked_at": datetime(next_year, next_month, 6, 18, 0),
                         "locked_by": "controller"})
        else:
            rows.append({"period": period, "close_status": "OPEN",
                         "locked_at": None, "locked_by": None})
    return pd.DataFrame(rows)


def build_database(db_path: str = DEFAULT_DB_PATH, raw_dir: str = RAW_DIR,
                   seed: int = SEED) -> str:
    rng = np.random.default_rng(seed)
    read = lambda name: pd.read_csv(os.path.join(raw_dir, name), dtype={"account_id": str})
    budget, actuals, forecast = read("budget.csv"), read("actuals.csv"), read("forecast.csv")
    with open(os.path.join(raw_dir, "gl_metadata.json")) as f:
        gl_meta = json.load(f)

    coa = pd.DataFrame([{"gl_code": g["account_id"], "account_name": g["name"],
                         "pl_category": g["type"], "normal_balance": g["normal"]}
                        for g in gl_meta])
    depts = pd.DataFrame([{"department_id": k, "department_name": v}
                          for k, v in DEPARTMENTS.items()])
    gl_entries = build_gl_entries(actuals, budget, rng)
    close_log = build_close_log()
    budget_targets = budget.rename(columns={"account_id": "gl_code"})[
        ["period", "gl_code", "department_id", "budget_amount"]]
    forecast_targets = forecast.rename(columns={"account_id": "gl_code"})[
        ["period", "gl_code", "department_id", "forecast_amount"]]

    # Build into a temp file and swap it in, so a reader never sees a half-built DB.
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    tmp_path = f"{db_path}.tmp-{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    con = duckdb.connect(tmp_path)
    try:
        create_schema(con)
        for table, frame in [("coa_mapping", coa), ("departments", depts),
                             ("close_control_log", close_log), ("gl_entries", gl_entries),
                             ("budget_targets", budget_targets),
                             ("forecast_targets", forecast_targets)]:
            con.register("frame", frame)
            cols = ", ".join(frame.columns)
            con.execute(f"INSERT INTO {table} ({cols}) SELECT {cols} FROM frame")
            con.unregister("frame")
    finally:
        con.close()
    os.replace(tmp_path, db_path)
    return db_path


def main():
    path = build_database()
    con = duckdb.connect(path, read_only=True)
    counts = con.execute("""
        SELECT entry_status, count(*) FROM gl_entries GROUP BY 1 ORDER BY 1
    """).fetchall()
    con.close()
    print(f"✅ Seeded {os.path.relpath(path, ROOT)}: "
          + ", ".join(f"{n} {status} entries" for status, n in counts))


if __name__ == "__main__":
    main()

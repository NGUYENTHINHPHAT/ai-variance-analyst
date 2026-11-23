-- ─────────────────────────────────────────────────────────────────────────────
-- Mock ERP schema (DuckDB)
--
-- Models the slice of a general ledger an FP&A team pulls at month-end:
--   coa_mapping        chart of accounts: GL code -> P&L category
--   departments        cost-center dimension
--   gl_entries         journal-entry lines (POSTED or DRAFT), one row per line
--   budget_targets     approved budget, one row per period x GL code x department
--   forecast_targets   latest rolling forecast, same grain as the budget
--   close_control_log  month-end close status per period (LOCKED / OPEN)
--
-- Deliberately has NO foreign key from gl_entries to coa_mapping/departments:
-- real ERP exports routinely contain codes the reporting mapping doesn't know
-- about yet, and catching them is the job of sql/checks/, not of the loader.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE coa_mapping (
    gl_code         VARCHAR PRIMARY KEY,
    account_name    VARCHAR NOT NULL,
    pl_category     VARCHAR NOT NULL CHECK (pl_category IN ('Revenue', 'COGS', 'OpEx')),
    normal_balance  VARCHAR NOT NULL CHECK (normal_balance IN ('debit', 'credit'))
);

CREATE TABLE departments (
    department_id    VARCHAR PRIMARY KEY,
    department_name  VARCHAR NOT NULL
);

CREATE TABLE close_control_log (
    period        VARCHAR PRIMARY KEY,          -- 'YYYY-MM'
    close_status  VARCHAR NOT NULL CHECK (close_status IN ('LOCKED', 'OPEN')),
    locked_at     TIMESTAMP,                    -- NULL while the period is OPEN
    locked_by     VARCHAR
);

CREATE TABLE gl_entries (
    entry_id       VARCHAR PRIMARY KEY,
    period         VARCHAR NOT NULL,
    gl_code        VARCHAR NOT NULL,
    department_id  VARCHAR NOT NULL,
    amount         DECIMAL(14, 2) NOT NULL,     -- natural sign for the account's normal balance
    entry_status   VARCHAR NOT NULL CHECK (entry_status IN ('POSTED', 'DRAFT')),
    description    VARCHAR
);

CREATE TABLE budget_targets (
    period         VARCHAR NOT NULL,
    gl_code        VARCHAR NOT NULL,
    department_id  VARCHAR NOT NULL,
    budget_amount  DECIMAL(14, 2) NOT NULL,
    PRIMARY KEY (period, gl_code, department_id)
);

CREATE TABLE forecast_targets (
    period           VARCHAR NOT NULL,
    gl_code          VARCHAR NOT NULL,
    department_id    VARCHAR NOT NULL,
    forecast_amount  DECIMAL(14, 2) NOT NULL,
    PRIMARY KEY (period, gl_code, department_id)
);

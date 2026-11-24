-- ─────────────────────────────────────────────────────────────────────────────
-- Variance base table: budget vs. actual vs. forecast
--
-- Grain: one row per period x GL code x department. This is the table the app
-- and the variance engine run on (it replaces reading erp_combined.csv).
--
-- Rules:
--   * Actuals count only POSTED entries in LOCKED periods. Drafts, and anything
--     in a period that is still OPEN, never reach the analysis.
--   * Only GL codes in coa_mapping are reported. Unmapped codes are dropped here
--     and surfaced by sql/checks/unmapped_gl_codes.sql instead of disappearing
--     silently.
--   * Zero / missing handling:
--       - locked period, budget line but nothing posted   -> actual = 0
--       - open period                                     -> actual = NULL
--       - posted actuals with no budget line              -> budget = 0
--       - budget = 0                                      -> variance_pct = NULL
--                                                            (never divide by zero)
-- ─────────────────────────────────────────────────────────────────────────────

WITH locked_periods AS (
    SELECT period
    FROM close_control_log
    WHERE close_status = 'LOCKED'
),

posted_actuals AS (
    SELECT
        e.period,
        e.gl_code,
        e.department_id,
        SUM(e.amount) AS actual_amount
    FROM gl_entries AS e
    JOIN locked_periods AS lp USING (period)
    WHERE e.entry_status = 'POSTED'
    GROUP BY e.period, e.gl_code, e.department_id
),

-- Every combination that has a budget, an actual, or both.
budget_vs_actual AS (
    SELECT
        COALESCE(b.period, a.period)               AS period,
        COALESCE(b.gl_code, a.gl_code)             AS gl_code,
        COALESCE(b.department_id, a.department_id) AS department_id,
        COALESCE(b.budget_amount, 0)               AS budget_amount,
        a.actual_amount
    FROM budget_targets AS b
    FULL OUTER JOIN posted_actuals AS a
        ON  a.period        = b.period
        AND a.gl_code       = b.gl_code
        AND a.department_id = b.department_id
),

reconciled AS (
    SELECT
        bva.period,
        bva.gl_code,
        bva.department_id,
        bva.budget_amount,
        CASE
            WHEN lp.period IS NULL THEN NULL                -- open period: no actuals yet
            ELSE COALESCE(bva.actual_amount, 0)             -- locked, nothing posted: zero
        END AS actual_amount,
        f.forecast_amount
    FROM budget_vs_actual AS bva
    LEFT JOIN locked_periods AS lp
        ON lp.period = bva.period
    LEFT JOIN forecast_targets AS f
        ON  f.period        = bva.period
        AND f.gl_code       = bva.gl_code
        AND f.department_id = bva.department_id
)

SELECT
    r.period,
    r.gl_code                                   AS account_id,
    c.account_name,
    c.pl_category                               AS account_type,
    r.department_id,
    d.department_name,
    CAST(r.budget_amount   AS DOUBLE)           AS budget_amount,
    CAST(r.actual_amount   AS DOUBLE)           AS actual_amount,
    CAST(r.forecast_amount AS DOUBLE)           AS forecast_amount,
    CAST(r.actual_amount - r.budget_amount AS DOUBLE)                  AS variance_amount,
    CAST((r.actual_amount - r.budget_amount)
         / NULLIF(r.budget_amount, 0) AS DOUBLE)                       AS variance_pct,
    ABS(CAST((r.actual_amount - r.budget_amount)
             / NULLIF(r.budget_amount, 0) AS DOUBLE))                  AS abs_variance_pct
FROM reconciled AS r
JOIN coa_mapping AS c                      -- inner join: unmapped codes are excluded
    ON c.gl_code = r.gl_code
LEFT JOIN departments AS d
    ON d.department_id = r.department_id
ORDER BY c.gl_code, r.period, r.department_id;

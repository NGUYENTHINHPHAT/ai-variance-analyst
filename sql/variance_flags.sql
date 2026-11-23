-- ─────────────────────────────────────────────────────────────────────────────
-- Variance flags: the detection logic of src/variance_engine.py, in SQL
--
-- Reads the `variance_base` table (the output of extract_variance_base.sql) and
-- returns one row per flagged period x account x department, ranked worst first.
-- tests/test_sql_layer.py checks that the output matches
-- variance_engine.build_variance_report() row for row.
--
-- Parameters (bound from the Python constants, so thresholds live in one place):
--   $rule_threshold   |variance %| that triggers a rule flag          (0.10)
--   $min_amount       minimum |variance $| for a rule flag            (5000)
--   $z_cutoff         |z-score| that counts as a statistical outlier  (2.0)
--   $min_stat_rows    rows an account needs before z-scores are used  (4)
--   $trend_threshold  per-period variance % that counts toward a trend (0.05)
--
-- When one row trips several methods it is reported once, with the first
-- method in this order: RULE_BASED, STATISTICAL, TREND.
-- ─────────────────────────────────────────────────────────────────────────────

WITH closed AS (
    SELECT *, COALESCE(variance_pct, 0) AS pct_or_zero
    FROM variance_base
    WHERE actual_amount IS NOT NULL
),

-- 1. Rule-based: big in percentage terms AND big enough in dollars to matter.
rule_flags AS (
    SELECT *, 'RULE_BASED' AS flag_type, 1 AS method_rank,
           CAST(NULL AS DOUBLE) AS z_score, CAST(NULL AS VARCHAR) AS trend_direction
    FROM closed
    WHERE abs_variance_pct >= $rule_threshold
      AND ABS(variance_amount) >= $min_amount
),

-- 2. Statistical: z-score of each row's variance % within its account,
--    across all departments and closed periods (population std dev).
z_scored AS (
    SELECT *,
           COUNT(*) OVER acct AS rows_in_account,
           ABS(pct_or_zero - AVG(pct_or_zero) OVER acct)
               / NULLIF(STDDEV_POP(pct_or_zero) OVER acct, 0) AS abs_z
    FROM closed
    WINDOW acct AS (PARTITION BY account_id)
),
stat_flags AS (
    SELECT * EXCLUDE (rows_in_account, abs_z),
           'STATISTICAL' AS flag_type, 2 AS method_rank,
           abs_z AS z_score, CAST(NULL AS VARCHAR) AS trend_direction
    FROM z_scored
    WHERE rows_in_account >= $min_stat_rows
      AND abs_z > $z_cutoff
),

-- 3. Trend: this period and the two before it all moved the same way by more
--    than the threshold (per account x department).
trend_window AS (
    SELECT *,
           LAG(pct_or_zero, 1) OVER series AS pct_prev_1,
           LAG(pct_or_zero, 2) OVER series AS pct_prev_2
    FROM closed
    WINDOW series AS (PARTITION BY account_id, department_id ORDER BY period)
),
trend_flags AS (
    SELECT * EXCLUDE (pct_prev_1, pct_prev_2),
           'TREND' AS flag_type, 3 AS method_rank,
           CAST(NULL AS DOUBLE) AS z_score,
           CASE WHEN pct_or_zero > 0 THEN 'OVER' ELSE 'UNDER' END AS trend_direction
    FROM trend_window
    WHERE pct_prev_2 IS NOT NULL
      AND (   (pct_or_zero >  $trend_threshold AND pct_prev_1 >  $trend_threshold AND pct_prev_2 >  $trend_threshold)
           OR (pct_or_zero < -$trend_threshold AND pct_prev_1 < -$trend_threshold AND pct_prev_2 < -$trend_threshold))
),

all_flags AS (
    SELECT * FROM rule_flags
    UNION ALL BY NAME SELECT * FROM stat_flags
    UNION ALL BY NAME SELECT * FROM trend_flags
),

-- One row per period x account x department: keep the highest-priority method.
deduplicated AS (
    SELECT *
    FROM all_flags
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY account_id, department_id, period
        ORDER BY method_rank
    ) = 1
),

scored AS (
    SELECT *,
           CASE
               WHEN abs_variance_pct >= 0.30 THEN 'CRITICAL'
               WHEN abs_variance_pct >= 0.20 THEN 'HIGH'
               WHEN abs_variance_pct >= 0.10 THEN 'MEDIUM'
               ELSE 'LOW'
           END AS severity
    FROM deduplicated
)

SELECT
    period, account_id, account_name, account_type, department_id, department_name,
    budget_amount, actual_amount, forecast_amount,
    variance_amount, variance_pct, abs_variance_pct,
    flag_type, z_score, trend_direction, severity,
    CASE
        WHEN account_type = 'Revenue' AND variance_pct > 0 THEN 'BEAT'
        WHEN account_type = 'Revenue' AND variance_pct < 0 THEN 'MISS'
        WHEN variance_pct > 0                              THEN 'OVER_BUDGET'
        ELSE 'UNDER_BUDGET'
    END AS direction
FROM scored
ORDER BY
    CASE severity WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END,
    abs_variance_pct DESC;

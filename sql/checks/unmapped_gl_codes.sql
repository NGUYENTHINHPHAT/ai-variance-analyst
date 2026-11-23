-- Posted entries in locked periods whose GL code is missing from coa_mapping.
-- extract_variance_base.sql inner-joins to coa_mapping, so these amounts would
-- otherwise drop out of the P&L without anyone noticing.
SELECT
    e.gl_code,
    COUNT(*)              AS entry_count,
    SUM(e.amount)         AS total_amount,
    MIN(e.period)         AS first_period,
    MAX(e.period)         AS last_period
FROM gl_entries AS e
JOIN close_control_log AS c
    ON c.period = e.period AND c.close_status = 'LOCKED'
LEFT JOIN coa_mapping AS m
    ON m.gl_code = e.gl_code
WHERE e.entry_status = 'POSTED'
  AND m.gl_code IS NULL
GROUP BY e.gl_code
ORDER BY ABS(SUM(e.amount)) DESC;

-- Mapped GL code x department combinations with posted actuals in a locked
-- period but no budget line. They are reported with budget = 0, so their
-- variance % is undefined (NULL) and the rule-based check can't flag them.
SELECT
    e.period,
    e.gl_code,
    e.department_id,
    SUM(e.amount)         AS actual_amount
FROM gl_entries AS e
JOIN close_control_log AS c
    ON c.period = e.period AND c.close_status = 'LOCKED'
JOIN coa_mapping AS m
    ON m.gl_code = e.gl_code
LEFT JOIN budget_targets AS b
    ON  b.period        = e.period
    AND b.gl_code       = e.gl_code
    AND b.department_id = e.department_id
WHERE e.entry_status = 'POSTED'
  AND b.gl_code IS NULL
GROUP BY e.period, e.gl_code, e.department_id
ORDER BY e.period, e.gl_code, e.department_id;

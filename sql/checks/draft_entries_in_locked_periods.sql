-- DRAFT entries still sitting in a period that has been closed and locked.
-- They are excluded from actuals; if any were meant to be posted, the close
-- missed them and the period's actuals are understated.
SELECT
    e.entry_id,
    e.period,
    e.gl_code,
    e.department_id,
    e.amount,
    e.description
FROM gl_entries AS e
JOIN close_control_log AS c
    ON c.period = e.period AND c.close_status = 'LOCKED'
WHERE e.entry_status = 'DRAFT'
ORDER BY e.period, e.entry_id;

-- Posted entries in locked periods booked to a department that doesn't exist.
-- They still count toward actuals, but show up with no department name.
SELECT
    e.department_id,
    COUNT(*)              AS entry_count,
    SUM(e.amount)         AS total_amount
FROM gl_entries AS e
JOIN close_control_log AS c
    ON c.period = e.period AND c.close_status = 'LOCKED'
LEFT JOIN departments AS d
    ON d.department_id = e.department_id
WHERE e.entry_status = 'POSTED'
  AND d.department_id IS NULL
GROUP BY e.department_id
ORDER BY entry_count DESC;

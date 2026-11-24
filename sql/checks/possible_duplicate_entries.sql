-- Posted lines with the same period, GL code, department and amount down to
-- the cent: the usual sign of a journal entry that was posted twice.
SELECT
    e.period,
    e.gl_code,
    e.department_id,
    e.amount,
    COUNT(*)                          AS times_posted,
    STRING_AGG(e.entry_id, ', ' ORDER BY e.entry_id) AS entry_ids
FROM gl_entries AS e
WHERE e.entry_status = 'POSTED'
GROUP BY e.period, e.gl_code, e.department_id, e.amount
HAVING COUNT(*) > 1
ORDER BY e.period, e.gl_code;

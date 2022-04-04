Find Minimum and maximum values of continuous sequence numbers in a group of rows?


SELECT dept_id, MIN(emp_seq) min_seq, MAX(emp_seq) max_seq
FROM
(
SELECT dept_id, emp_seq,emp_seq - ROW_NUMBER() OVER (PARTITION BY dept_id ORDER BY emp_seq) dept_split
FROM DATA
) A
GROUP BY dept_id, dept_split order by 1
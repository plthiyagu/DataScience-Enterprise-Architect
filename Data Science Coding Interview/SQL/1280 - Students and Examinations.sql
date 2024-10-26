# Write your MySQL query statement below
SELECT 
  s.student_id, 
  s.student_name, 
  s.subject_name, 
  COUNT(e.subject_name) AS attended_exams 
FROM 
  (
    SELECT 
      * 
    FROM 
      Students s CROSS 
      JOIN Subjects su
  ) s 
  LEFT JOIN Examinations e ON s.student_id = e.student_id 
  AND s.subject_name = e.subject_name 
GROUP BY 
  s.student_id, 
  s.subject_name 
ORDER BY 
  s.student_id, 
  s.subject_name

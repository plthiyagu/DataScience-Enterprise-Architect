SELECT 
  u.unique_id, 
  e.name 
FROM 
  Employees AS e 
  LEFT JOIN EmployeeUNI AS u USING(id)
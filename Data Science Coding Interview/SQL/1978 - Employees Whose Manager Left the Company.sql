SELECT 
  employee_id 
FROM 
  Employees 
WHERE 
  salary < 30000 
  AND manager_id NOT IN (
    SELECT 
      DISTINCT(employee_id) 
    FROM 
      Employees
  ) 
ORDER BY 
  employee_id ASC

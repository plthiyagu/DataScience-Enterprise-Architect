SELECT 
  r.Employee, 
  r.Department, 
  r.Salary 
FROM 
  (
    SELECT 
      d.name AS Department, 
      e.name AS Employee, 
      e.salary AS Salary, 
      DENSE_RANK() OVER(
        PARTITION BY departmentId 
        ORDER BY 
          salary DESC
      ) AS top_rank 
    FROM 
      Employee e 
      INNER JOIN Department d ON e.departmentId = d.id
  ) AS r 
WHERE 
  r.top_rank <= 3

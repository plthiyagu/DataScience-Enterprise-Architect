SELECT 
  dr.name 
FROM 
  (
    SELECT 
      e.id, 
      e.name, 
      COUNT(m.id) AS count_employees 
    FROM 
      Employee AS e 
      INNER JOIN Employee AS m ON e.id = m.managerId 
    GROUP BY 
      e.id
  ) AS dr 
WHERE 
  dr.count_employees >= 5

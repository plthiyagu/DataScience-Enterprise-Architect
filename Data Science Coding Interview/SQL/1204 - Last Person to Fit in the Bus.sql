SELECT 
  person_name 
FROM 
  (
    SELECT 
      person_name, 
      turn, 
      SUM(weight) over (
        ORDER BY 
          turn
      ) AS cum 
    FROM 
      queue
  ) p1 
WHERE 
  cum <= 1000 
ORDER BY 
  turn DESC 
limit 
  1;

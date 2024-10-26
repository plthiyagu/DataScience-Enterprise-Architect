SELECT 
  pm.machine_id, 
  ROUND(
    AVG(pm.end - pm.start), 
    3
  ) AS processing_time 
FROM 
  (
    SELECT 
      a1.machine_id, 
      a1.process_id, 
      a1.timestamp AS end, 
      a2.timestamp AS start 
    FROM 
      Activity a1 
      INNER JOIN Activity a2 ON a1.machine_id = a2.machine_id 
      AND a1.process_id = a2.process_id 
      AND a1.timestamp > a2.timestamp
  ) AS pm 
GROUP BY 
  pm.machine_id
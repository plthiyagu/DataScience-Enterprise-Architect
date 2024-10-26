SELECT 
  cr.user_id, 
  ROUND(
    SUM(
      CASE WHEN cr.action = 'confirmed' THEN 1 ELSE 0 END
    ) / COUNT(cr.user_id), 
    2
  ) AS confirmation_rate 
FROM 
  (
    SELECT 
      s.user_id, 
      s.time_stamp, 
      c.action 
    FROM 
      Signups s 
      LEFT JOIN Confirmations c ON s.user_id = c.user_id
  ) AS cr 
GROUP BY 
  cr.user_id
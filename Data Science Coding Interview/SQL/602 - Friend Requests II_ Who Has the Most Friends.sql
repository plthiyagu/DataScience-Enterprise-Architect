SELECT 
  r.id, 
  COUNT(r.id) AS num 
FROM 
  (
    (
      SELECT 
        requester_id AS id 
      FROM 
        RequestAccepted
    ) 
    UNION ALL 
      (
        SELECT 
          accepter_id AS id 
        FROM 
          RequestAccepted
      )
  ) AS r 
GROUP BY 
  r.id 
ORDER BY 
  num DESC 
LIMIT 
  1

SELECT 
  ROUND(
    COUNT(DISTINCT a1.player_id)/(
      SELECT 
        COUNT(DISTINCT player_id) 
      FROM 
        Activity
    ), 
    2
  ) AS fraction 
FROM 
  Activity a1 
  INNER JOIN (
    SELECT 
      player_id, 
      min(event_date) AS first_date 
    FROM 
      Activity 
    GROUP BY 
      player_id
  ) a2 ON a1.player_id = a2.player_id 
  AND a1.event_date = ADDDATE(a2.first_date, INTERVAL 1 DAY)

    SELECT user_id,
       occurred_at,
       SUM(is_new_session) OVER (ORDER BY user_id, occurred_at) AS global_session_id,
       SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY occurred_at) AS user_session_id
      FROM (
        SELECT *,
              CASE WHEN EXTRACT('EPOCH' FROM occurred_at) - EXTRACT('EPOCH' FROM last_event) >= (60 * 10) 
                     OR last_event IS NULL 
                   THEN 1 ELSE 0 END AS is_new_session
         FROM (
              SELECT user_id,
                     occurred_at,
                     LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) AS last_event
                FROM tutorial.playbook_events
              ) last
       ) final
    LIMIT 100

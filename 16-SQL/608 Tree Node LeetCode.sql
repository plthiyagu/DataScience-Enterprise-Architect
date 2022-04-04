SELECT id, 
CASE WHEN p_id is null THEN 'Root'
WHEN id not in 
(SELECT p_id FROM Tree WHERE p_id is not null GROUP BY p_id) THEN 'Leaf'
ELSE 'Inner' END AS Type FROM Tree order by id

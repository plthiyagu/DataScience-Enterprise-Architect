SELECT t.id, COUNT(t.id) AS num FROM
(SELECT requester_id AS id FROM request_accepted
 UNION ALL
 SELECT accepter_id AS id FROM request_accepted) AS t
GROUP BY t.id
ORDER BY num DESC LIMIT 1;

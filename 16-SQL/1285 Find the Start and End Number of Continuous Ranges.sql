# Write your MySQL query statement below

SELECT MIN(log_id) AS start_id, MAX(log_id) AS end_id
FROM
(SELECT l1.log_id,(l1.log_id- l1.row_num) AS diff 
 FROM
(SELECT log_id, ROW_NUMBER() OVER() AS row_num  FROM logs) l1
 )l2
GROUP BY diff
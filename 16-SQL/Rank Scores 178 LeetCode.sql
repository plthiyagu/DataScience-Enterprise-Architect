# Write your MySQL query statement below
select score,
    (select count(distinct score) + 1 
     from scores ss 
     where ss.score > s.score) 'rank'
from scores s
order by score desc

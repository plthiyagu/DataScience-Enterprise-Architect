# Write your MySQL query statement below
SELECT DISTINCT l1.Num AS ConsecutiveNums FROM
 logs l1, logs l2, logs l3
 where (l1.num= l2.num and l2.num=l3.num
             and l1.Id = l2.Id + 1 and 
      l2.Id = l3.Id + 1)
group by l1.Num


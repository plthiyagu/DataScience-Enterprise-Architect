# Write your MySQL query statement below
select d.Name as Department,
       e1.Name as Employee,
       e1.salary as Salary
FROM Employee e1
JOIN Department d ON e1.DepartmentId = d.Id 

where 3 > (select count(distinct Salary)
       from Employee 
       where Salary >e1.Salary
       and DepartmentId = e1.DepartmentId)

CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  RETURN (
      # Write your MySQL query statement below.
      select distinct Salary as 'getNthHighestSalary(2)'
      from(
      select Salary, 
          dense_rank() over (order by Salary desc) as rnk
          from Employee
          ) sal_rnk
      where rnk=n    
  );
END
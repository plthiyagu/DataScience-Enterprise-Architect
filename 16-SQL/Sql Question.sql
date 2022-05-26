Write a SQl query to get a column calculation containing (calculation = revenue / sum_of_sales) and Sum_of_sales containing (Sum_of_sales = Addition of sales with above all sales partition by ID )
Go through the attachment for preferred i/p & o/p


with cte as 
(
select * , row_number () over (partition by id order by (select null)) as row_num from #temp_)
, CTE_2 as 
(
SELECT *, SUM(sales) over (partition by id order by row_num) as sum_of_sales FROM cte t2 
), CTE_3 as 
(
select id,sales,sum_of_sales,revenue, revenue/sum_of_sales as calculation from cte_2
)

select * from cte_3
-- Create your SELECT statement here

select age, count(*) as total_people
from people
group by age
having count(*) >= 10





SELECT age,count(age) AS total_people from people 

group by age 

having count(age) >=10
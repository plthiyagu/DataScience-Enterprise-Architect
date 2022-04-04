select store, 'Q' + cast(10-sum(cast(right(quarter,1) as int)) as char(2)) as q_no 
from stores
group by store;
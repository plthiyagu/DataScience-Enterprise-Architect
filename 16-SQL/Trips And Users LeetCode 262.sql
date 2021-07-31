 select t.Request_at as Day, round(1 - sum(t.status = "Completed") / t.all_count, 2) as "Cancellation Rate"
from (select *, count(status) over (partition by request_at) as all_count
     from Trips
     where client_id in (select users_id
                        from Users
                        where banned = "No") and
            driver_id in (select users_id
                        from Users
                        where banned = "No")) as t
where t.request_at in ('2013-10-01','2013-10-02','2013-10-03') 
group by t.request_at

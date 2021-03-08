use employees;

CREATE TABLE web ( ts timestamp,
						userid int,
                        web_sessionid int);
CREATE TABLE iphone ( ts timestamp,
						userid int,
                        web_sessionid int);
INSERT into web (ts,userid,web_sessionid)
VALUES 
     (now(), 1, 100),
     (now() - INTERVAL 1 HOUR, 1,101),
	 (now() - INTERVAL 1 day, 2,103);

     
INSERT into iphone (ts,userid,web_sessionid)
VALUES 
     (now(), 1, 100),
     (now() -INTERVAL 2 HOUR,1,101),
	 (now() -INTERVAL 1 day, 3,103);
     
     
     
SELECT DATE(i.ts) as day,
count(distinct i.userid) as num_users
FROM  iphone i 
JOIN web w on i.userid =w.userid
AND DATE(i.ts) = DATE(w.ts)
group by 1



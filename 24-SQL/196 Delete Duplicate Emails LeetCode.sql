# Write your MySQL query statement below

DELETE p1 from Person p1,Person p2
  where p1.Email = p2.Email and p1.id >p2.id

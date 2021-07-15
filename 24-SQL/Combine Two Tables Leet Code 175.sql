#175. Combine Two Tables

# Write your MySQL query statement below

select Person.FirstName,Person.Lastname, Address.City,Address.State
from  Person 
left join Address on Person.PersonId =Address.PersonId;

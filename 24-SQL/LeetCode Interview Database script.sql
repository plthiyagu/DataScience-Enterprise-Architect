show databases;
use interview;
show tables;




create table Person(
   PersonId INT NOT NULL AUTO_INCREMENT,
   FirstName VARCHAR(40),
   LastName VARCHAR(40),
   PRIMARY KEY (PersonId)
);

create table Address(
   AddressId INT NOT NULL AUTO_INCREMENT,
   PersonId INT,
   City VARCHAR(40),
   State VARCHAR(40),
   PRIMARY KEY ( AddressId),
   FOREIGN KEY (PersonId) REFERENCES Person(PersonId)
);

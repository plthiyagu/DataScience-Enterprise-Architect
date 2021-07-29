-- Create your SELECT statement here


SELECT d.id, d.name FROM departments d 
WHERE EXISTS(SELECT * 
            FROM sales s WHERE s.department_id = d.id AND s.price >98)
            
            

SELECT d.* 
FROM departments d
WHERE EXISTS (SELECT 1 FROM sales s WHERE s.price>98 and s.department_id=d.id);

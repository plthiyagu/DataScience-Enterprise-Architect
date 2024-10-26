SELECT 
  product_name, 
  year, 
  price 
FROM 
  Sales AS s 
  INNER JOIN Product p ON p.product_id = s.product_id

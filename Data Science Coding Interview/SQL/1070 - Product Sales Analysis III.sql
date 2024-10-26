SELECT 
  s.product_id, 
  my.first_year, 
  s.quantity, 
  s.price 
FROM 
  Sales s 
  INNER JOIN (
    SELECT 
      product_id, 
      MIN(year) AS first_year 
    FROM 
      Sales 
    GROUP BY 
      product_id
  ) AS my 
WHERE 
  s.year = my.first_year 
  AND s.product_id = my.product_id

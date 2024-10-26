SELECT 
  CONCAT_WS(
    '-', 
    YEAR(trans_date), 
    LPAD(
      MONTH(trans_date), 
      2, 
      0
    )
  ) AS month, 
  country, 
  COUNT(DISTINCT id) AS trans_count, 
  SUM(state = 'approved') AS approved_count, 
  SUM(amount) AS trans_total_amount, 
  SUM(
    (state = 'approved') * amount
  ) AS approved_total_amount 
FROM 
  Transactions 
GROUP BY 
  month, 
  country

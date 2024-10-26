SELECT 
  ROUND(
    (
      SELECT 
        COUNT(DISTINCT d.customer_id) 
      FROM 
        Delivery AS d 
        INNER JOIN (
          SELECT 
            customer_id, 
            min(order_date) AS first_order 
          FROM 
            Delivery 
          GROUP BY 
            customer_id
        ) AS fo ON d.customer_id = fo.customer_id 
        AND d.order_date = fo.first_order 
      WHERE 
        fo.first_order = d.customer_pref_delivery_date
    )/ COUNT(DISTINCT customer_id) * 100, 
    2
  ) AS immediate_percentage 
FROM 
  Delivery

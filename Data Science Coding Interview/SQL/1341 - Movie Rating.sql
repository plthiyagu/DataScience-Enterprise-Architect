(
  SELECT 
    u.name AS results 
  FROM 
    Users u 
    INNER JOIN MovieRating mr ON u.user_id = mr.user_id 
  GROUP BY 
    u.user_id 
  ORDER BY 
    COUNT(DISTINCT mr.movie_id) DESC, 
    u.name ASC 
  LIMIT 
    1
) 
UNION ALL 
  (
    SELECT 
      m.title AS results 
    FROM 
      Movies m 
      INNER JOIN MovieRating mr ON m.movie_id = mr.movie_id 
    WHERE 
      MONTH(mr.created_at) = 2 
      AND YEAR(mr.created_at) = 2020 
    GROUP BY 
      m.movie_id 
    ORDER BY 
      AVG(mr.rating) DESC, 
      m.title ASC 
    LIMIT 
      1
  )

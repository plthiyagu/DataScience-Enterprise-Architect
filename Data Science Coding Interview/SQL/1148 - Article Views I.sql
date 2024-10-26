SELECT 
  DISTINCT(author_id) AS id 
from 
  Views 
WHERE 
  author_id = viewer_id 
ORDER BY 
  author_id ASC
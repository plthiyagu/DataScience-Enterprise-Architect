--Write your SQL statement here--
SELECT (CASE WHEN (number%2) <> 0  THEN 'Odd'
         ELSE 'Even'  End) is_even FROM numbers 
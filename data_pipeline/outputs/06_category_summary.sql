SELECT c.category_name,
               COUNT(*) AS book_count,
               ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
        FROM books AS b
        JOIN categories AS c
          ON b.category_id = c.category_id
        GROUP BY c.category_id, c.category_name
        ORDER BY c.category_name;

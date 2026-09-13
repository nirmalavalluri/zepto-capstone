SELECT title, rating, price_gbp
        FROM books
        ORDER BY rating DESC, price_gbp, book_id
        LIMIT 10;

SELECT title, price_gbp
        FROM books
        WHERE price_gbp < 20
        ORDER BY price_gbp, title, book_id;

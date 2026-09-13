from pathlib import Path
from urllib.parse import urljoin
import json
import sqlite3
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://books.toscrape.com/"
GBP_TO_INR = 105.50
CATEGORIES = ("Travel", "Mystery", "Historical Fiction")
RATINGS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

DATABASE = OUTPUT / "books.db"


def get_page(session, url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    response.encoding = "utf-8"
    return BeautifulSoup(response.text, "html.parser")


def scrape_books():
    records = []
    seen_urls = set()

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "CapstoneEducationalScraper/1.0"
        })

        home = get_page(session, BASE_URL)
        category_urls = {}

        for link in home.select(".side_categories a"):
            name = link.get_text(strip=True)
            if name in CATEGORIES:
                category_urls[name] = urljoin(BASE_URL, link["href"])

        missing = set(CATEGORIES) - set(category_urls)
        if missing:
            raise RuntimeError(f"Categories not found: {sorted(missing)}")

        for category in CATEGORIES:
            page_url = category_urls[category]
            print(f"Scraping {category}...")

            while page_url:
                page = get_page(session, page_url)

                for book in page.select("article.product_pod"):
                    title_link = book.select_one("h3 a")
                    price = book.select_one(".price_color")
                    rating = book.select_one(".star-rating")
                    availability = book.select_one(".availability")

                    # Retain malformed fields as missing for cleaning.
                    title = (
                        title_link.get("title")
                        or title_link.get_text(strip=True)
                        if title_link else None
                    )

                    book_url = (
                        urljoin(page_url, title_link["href"])
                        if title_link and title_link.get("href")
                        else None
                    )

                    if book_url and book_url in seen_urls:
                        continue
                    if book_url:
                        seen_urls.add(book_url)

                    rating_text = next(
                        (
                            value
                            for value in rating.get("class", [])
                            if value in RATINGS
                        ),
                        None,
                    ) if rating else None

                    records.append({
                        "title": title,
                        "price": price.get_text(strip=True) if price else None,
                        "star_rating": rating_text,
                        "availability": (
                            availability.get_text(" ", strip=True)
                            if availability else None
                        ),
                        "category": category,
                        "source_url": book_url,
                    })

                next_link = page.select_one("li.next a")
                page_url = (
                    urljoin(page_url, next_link["href"])
                    if next_link else None
                )
                time.sleep(0.3)

    if not records:
        raise RuntimeError("The scraper returned no books.")

    return pd.DataFrame(records)


def parse_availability(value):
    if not isinstance(value, str):
        return None

    text = value.strip().lower()

    # Check negative wording first.
    if "out of stock" in text or "not in stock" in text:
        return False
    if "in stock" in text:
        return True

    return None


def clean_books(raw):
    df = raw.copy()

    df["price_gbp"] = pd.to_numeric(
        df["price"].str.replace("£", "", regex=False).str.strip(),
        errors="coerce",
    )
    df["rating"] = df["star_rating"].map(RATINGS)
    df["in_stock"] = df["availability"].apply(parse_availability)

    valid = (
        df["title"].fillna("").str.strip().ne("")
        & df["category"].isin(CATEGORIES)
        & df["price_gbp"].notna()
        & df["price_gbp"].ge(0)
        & df["rating"].between(1, 5)
        & df["in_stock"].notna()
        & df["source_url"].notna()
    )

    rejected = df.loc[~valid].copy()
    rejected.to_csv(OUTPUT / "rejected_books.csv", index=False)

    df = df.loc[valid].copy()
    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)
    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR).round(2)

    if len(df) < 60 or df["category"].nunique() < 3:
        raise RuntimeError(
            f"Insufficient cleaned data: {len(df)} books, "
            f"{df['category'].nunique()} categories."
        )

    df.insert(0, "book_id", range(1, len(df) + 1))

    report = {
        "raw_rows": len(raw),
        "clean_rows": len(df),
        "rejected_rows": len(rejected),
        "category_count": int(df["category"].nunique()),
        "gbp_to_inr": GBP_TO_INR,
        "cleaning_policy": (
            "Drop rows with missing or invalid required fields; "
            "save rejected rows for inspection."
        ),
    }

    (OUTPUT / "cleaning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    return df


def create_tables(df):
    categories = pd.DataFrame({
        "category_name": sorted(df["category"].unique())
    })
    categories.insert(
        0, "category_id", range(1, len(categories) + 1)
    )

    books = df.merge(
        categories,
        left_on="category",
        right_on="category_name",
        validate="many_to_one",
    )[
        [
            "book_id", "title", "price_gbp", "price_inr",
            "rating", "in_stock", "category_id",
        ]
    ].copy()

    return categories, books


QUERIES = {
    "01_affordable_books": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp < 20
        ORDER BY price_gbp, title, book_id;
    """,
    "02_top_rated_books": """
        SELECT title, rating, price_gbp
        FROM books
        ORDER BY rating DESC, price_gbp, book_id
        LIMIT 10;
    """,
    "03_distinct_ratings": """
        SELECT DISTINCT rating
        FROM books
        ORDER BY rating;
    """,
    "04_price_range": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 30
        ORDER BY price_gbp, title, book_id;
    """,
    "05_books_with_categories": """
        SELECT b.book_id, b.title, c.category_name,
               b.price_gbp, b.price_inr, b.rating, b.in_stock
        FROM books AS b
        JOIN categories AS c
          ON b.category_id = c.category_id
        ORDER BY b.book_id;
    """,
    "06_category_summary": """
        SELECT c.category_name,
               COUNT(*) AS book_count,
               ROUND(AVG(b.price_gbp), 2) AS average_price_gbp
        FROM books AS b
        JOIN categories AS c
          ON b.category_id = c.category_id
        GROUP BY c.category_id, c.category_name
        ORDER BY c.category_name;
    """,
}


def save_database_and_queries(categories, books):
    with sqlite3.connect(DATABASE) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        connection.executescript("""
            DROP TABLE IF EXISTS books;
            DROP TABLE IF EXISTS categories;

            CREATE TABLE categories (
                category_id INTEGER PRIMARY KEY,
                category_name TEXT NOT NULL UNIQUE
            );

            CREATE TABLE books (
                book_id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                price_gbp REAL NOT NULL CHECK(price_gbp >= 0),
                price_inr REAL NOT NULL CHECK(price_inr >= 0),
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                in_stock INTEGER NOT NULL CHECK(in_stock IN (0, 1)),
                category_id INTEGER NOT NULL,
                FOREIGN KEY(category_id) REFERENCES categories(category_id)
            );
        """)

        connection.executemany(
            "INSERT INTO categories VALUES (?, ?)",
            categories.itertuples(index=False, name=None),
        )

        database_books = books.copy()
        database_books["in_stock"] = (
            database_books["in_stock"].astype(int)
        )

        connection.executemany(
            "INSERT INTO books VALUES (?, ?, ?, ?, ?, ?, ?)",
            database_books.itertuples(index=False, name=None),
        )
        connection.commit()

        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()
        if foreign_key_errors:
            raise RuntimeError(str(foreign_key_errors))

        results = {}
        transcript = []

        for name, sql in QUERIES.items():
            result = pd.read_sql(sql, connection)
            results[name] = result

            (OUTPUT / f"{name}.sql").write_text(
                sql.strip() + "\n", encoding="utf-8"
            )
            result.to_csv(OUTPUT / f"{name}.csv", index=False)

            transcript.append(
                f"{name}\n{sql.strip()}\n\n"
                f"{result.to_string(index=False)}\n"
            )

        (OUTPUT / "query_results.txt").write_text(
            "\n\n".join(transcript), encoding="utf-8"
        )

    # Reproduce the SQL join directly from in-memory DataFrames.
    pandas_join = pd.merge(
        books, categories, on="category_id", validate="many_to_one"
    )

    sql_join = results["05_books_with_categories"]

    pandas_join = (
        pandas_join[sql_join.columns]
        .sort_values("book_id")
        .reset_index(drop=True)
    )

    # SQLite represents booleans as integers.
    pandas_join["in_stock"] = pandas_join["in_stock"].astype(int)

    pd.testing.assert_frame_equal(
        sql_join, pandas_join, check_dtype=False
    )

    comparison = pd.concat(
        {"SQL": sql_join, "pandas_merge": pandas_join},
        axis=1,
    )
    comparison.to_csv(OUTPUT / "join_comparison.csv", index=False)

    (OUTPUT / "join_comparison.txt").write_text(
        "Full SQL and pandas join results match.\n\n"
        + comparison.to_string(index=False),
        encoding="utf-8",
    )

    print("\nCategory summary:")
    print(results["06_category_summary"].to_string(index=False))
    print("\nPASS: SQL JOIN and pandas merge results match.")


def main():
    raw = scrape_books()
    raw.to_csv(OUTPUT / "raw_books.csv", index=False)

    cleaned = clean_books(raw)
    cleaned.to_csv(OUTPUT / "cleaned_books.csv", index=False)

    categories, books = create_tables(cleaned)
    save_database_and_queries(categories, books)

    print(f"\nCleaned books: {len(cleaned)}")
    print(f"Categories: {cleaned['category'].nunique()}")
    print(f"Database: {DATABASE}")
    print(f"Outputs: {OUTPUT}")
    print("\nModule 1 pipeline completed successfully.")


if __name__ == "__main__":
    main()
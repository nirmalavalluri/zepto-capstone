# Module 1 Data Pipeline

## Setup and run
From the repository root:
python -m pip install -r data_pipeline/requirements.txt
python data_pipeline/main.py

Internet access is required for scraping.

## Source and results
Source: https://books.toscrape.com/

The scraper follows category pagination and collected 69 cleaned books:
- Travel: 11 books, average price GBP 39.79.
- Mystery: 32 books, average price GBP 31.72.
- Historical Fiction: 26 books, average price GBP 33.64.

## Cleaning decisions
Currency symbols are removed and prices are converted to floats.
Text ratings are mapped to integers from 1 to 5.
Availability is converted to boolean, checking negative wording first.
Duplicate source URLs are excluded.

Rows with invalid or missing required fields are dropped to avoid
inventing prices, ratings, or availability. Rejected rows are saved
in outputs/rejected_books.csv. Counts are recorded in
outputs/cleaning_report.json.

The conversion uses exactly 1 GBP = 105.50 INR.
This is the fixed project-defined constant, not a market exchange rate.
INR prices are rounded to two decimal places.

## SQLite schema
categories: category_id primary key, category_name unique.
books: book_id primary key, category_id foreign key referencing categories.

Foreign-key enforcement is enabled during loading.
SQLite stores availability as 0 or 1; pandas uses boolean values.
Running main.py recreates these tables from freshly scraped data.

## Queries and pandas verification
Six saved SQL queries demonstrate SELECT, WHERE, ORDER BY, LIMIT,
DISTINCT, BETWEEN, JOIN, and GROUP BY with COUNT and AVG.

All six query results are read into DataFrames using pandas.read_sql.
The books/categories join is independently reproduced using pandas.merge
on the in-memory DataFrames.

After normalizing boolean representation, assert_frame_equal verifies
that the full SQL and pandas results match. The successful run passed.

## Output files
The outputs folder contains:
- raw_books.csv and cleaned_books.csv.
- rejected_books.csv and cleaning_report.json.
- books.db.
- Six numbered SQL files and their CSV results.
- query_results.txt containing query strings and full outputs.
- join_comparison.csv and join_comparison.txt showing SQL and pandas
  results side by side.

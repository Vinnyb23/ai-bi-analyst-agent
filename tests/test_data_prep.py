import os

from src.data_prep import generate_sales_data, load_to_sqlite, get_schema_description


def test_generate_sales_data_has_expected_columns():
    df = generate_sales_data(start="2024-01-01", end="2024-01-31")
    expected_cols = {
        "order_id", "order_date", "region", "category", "sub_category",
        "quantity", "unit_price", "discount", "sales", "profit",
    }
    assert expected_cols.issubset(set(df.columns))
    assert len(df) > 0


def test_generate_sales_data_is_deterministic_with_seed():
    df1 = generate_sales_data(start="2024-01-01", end="2024-01-10", seed=1)
    df2 = generate_sales_data(start="2024-01-01", end="2024-01-10", seed=1)
    assert df1.equals(df2)


def test_load_to_sqlite_creates_queryable_db(tmp_path):
    df = generate_sales_data(start="2024-01-01", end="2024-01-05")
    db_path = str(tmp_path / "test.db")
    load_to_sqlite(df, db_path=db_path)
    assert os.path.exists(db_path)

    import sqlite3
    conn = sqlite3.connect(db_path)
    count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    conn.close()
    assert count == len(df)


def test_get_schema_description_mentions_sales_table():
    desc = get_schema_description()
    assert "sales" in desc.lower()
    assert "region" in desc.lower()

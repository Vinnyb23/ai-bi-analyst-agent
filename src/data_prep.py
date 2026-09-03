"""
data_prep.py
------------
Generates a synthetic retail-sales dataset and loads it into a local SQLite
database (data/bi_analyst.db) -- this is a fresh, standalone copy of the
same "sales" schema used in Phase 1 (bi-ai-assistant), so this repo has zero
external dependencies and can be run/deployed on its own, while still
representing "your Phase 1 BI database" conceptually for the Week 15 agent
pipeline (src/agents.py queries this table).

Usage:
    python -m src.data_prep
"""

import os
import sqlite3

import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bi_analyst.db")
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
CATEGORIES = ["Furniture", "Office Supplies", "Technology"]
SUB_CATEGORIES = {
    "Furniture": ["Chairs", "Tables", "Bookcases"],
    "Office Supplies": ["Paper", "Binders", "Storage"],
    "Technology": ["Phones", "Accessories", "Machines"],
}

SCHEMA_DESCRIPTION = """\
Table: sales
Columns:
  order_id     INTEGER  -- unique order identifier
  order_date   TEXT     -- ISO date (YYYY-MM-DD)
  region       TEXT     -- one of: Northeast, Southeast, Midwest, West, Southwest
  category     TEXT     -- one of: Furniture, Office Supplies, Technology
  sub_category TEXT     -- e.g. Chairs, Paper, Phones
  quantity     INTEGER
  unit_price   REAL
  discount     REAL     -- fraction, e.g. 0.15
  sales        REAL     -- unit_price * quantity * (1 - discount)
  profit       REAL
"""


def generate_sales_data(start: str = "2023-01-01", end: str = "2026-08-01", seed: int = 42) -> pd.DataFrame:
    """Daily-grain synthetic sales table with weekday/seasonal/growth patterns + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, end=end, freq="D")

    rows = []
    order_id = 200000
    for date in dates:
        day_factor = 1.3 if date.weekday() < 5 else 0.8
        season_factor = 1.6 if date.month in (11, 12) else 1.0
        growth_factor = 1 + 0.15 * ((date.year - 2023) + date.month / 12)
        n_orders = int(rng.poisson(lam=6 * day_factor * season_factor * growth_factor))

        for _ in range(n_orders):
            region = rng.choice(REGIONS)
            category = rng.choice(CATEGORIES)
            sub_category = rng.choice(SUB_CATEGORIES[category])
            base_price = {"Furniture": 220, "Office Supplies": 35, "Technology": 310}[category]
            quantity = int(rng.integers(1, 6))
            unit_price = max(5, rng.normal(base_price, base_price * 0.25))
            discount = float(rng.choice([0, 0, 0.1, 0.15, 0.2], p=[0.5, 0.2, 0.15, 0.1, 0.05]))
            sales = round(unit_price * quantity * (1 - discount), 2)
            profit = round(sales * rng.normal(0.18, 0.08), 2)

            order_id += 1
            rows.append(
                {
                    "order_id": order_id,
                    "order_date": date.strftime("%Y-%m-%d"),
                    "region": region,
                    "category": category,
                    "sub_category": sub_category,
                    "quantity": quantity,
                    "unit_price": round(unit_price, 2),
                    "discount": discount,
                    "sales": sales,
                    "profit": profit,
                }
            )

    return pd.DataFrame(rows)


def load_to_sqlite(df: pd.DataFrame, db_path: str = DB_PATH) -> None:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("sales", conn, if_exists="replace", index=False)
    finally:
        conn.close()


def get_schema_description() -> str:
    return SCHEMA_DESCRIPTION


def ensure_database(db_path: str = DB_PATH) -> str:
    """Builds the DB the first time this is called; reuses it after that."""
    if not os.path.exists(db_path):
        df = generate_sales_data()
        load_to_sqlite(df, db_path)
    return db_path


if __name__ == "__main__":
    df = generate_sales_data()
    load_to_sqlite(df)
    print(f"Wrote {len(df)} rows to {DB_PATH}")

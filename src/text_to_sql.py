"""
text_to_sql.py
--------------
Turns a plain-English question into a read-only SQL query against the
`sales` table, executes it safely, and returns a DataFrame. Same
provider-agnostic pattern as Phase 1 (bi-ai-assistant): OpenAI or Anthropic,
selected via LLM_PROVIDER / LLM_MODEL in .env.

Zero-key fallback: if no API key is configured, `ask()` matches the
question against a small set of canned queries (FALLBACK_QUERIES) instead
of raising, so the rest of the pipeline (agents.py, app.py, tests) keeps
working without any credentials.

Safety: only single SELECT statements are allowed to run against the DB.
Anything else (INSERT/UPDATE/DELETE/DROP/ATTACH/PRAGMA...) is rejected
before execution -- never trust generated SQL blindly.

Usage:
    from src.text_to_sql import ask
    df, sql, used_llm = ask("What were total sales by region last quarter?")
"""

import os
import re
import sqlite3

import pandas as pd
from dotenv import load_dotenv

from src.data_prep import DB_PATH, ensure_database, get_schema_description

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # openai | anthropic
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are a SQL generator for a SQLite database. Given a user question and \
the schema below, output ONLY a single valid, read-only SQLite SELECT statement that answers \
the question. Never modify data. Never use more than one statement. Do not wrap the SQL in \
markdown code fences -- output raw SQL only.

Schema:
{schema}
"""

_DISALLOWED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|PRAGMA|CREATE|REPLACE|VACUUM)\b",
    re.IGNORECASE,
)

# A few canned fallback queries so the pipeline has something real to run
# even before an API key is configured.
FALLBACK_QUERIES = {
    "total sales by region": "SELECT region, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY region ORDER BY total_sales DESC;",
    "total sales by category": "SELECT category, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY category ORDER BY total_sales DESC;",
    "monthly sales trend": "SELECT strftime('%Y-%m', order_date) AS month, ROUND(SUM(sales), 2) AS total_sales FROM sales GROUP BY month ORDER BY month;",
    "top sub-categories by profit": "SELECT sub_category, ROUND(SUM(profit), 2) AS total_profit FROM sales GROUP BY sub_category ORDER BY total_profit DESC LIMIT 10;",
    "average discount by region": "SELECT region, ROUND(AVG(discount), 3) AS avg_discount FROM sales GROUP BY region ORDER BY avg_discount DESC;",
}


class UnsafeSQLError(Exception):
    pass


def _validate_sql(sql: str) -> str:
    sql = sql.strip().strip("`").strip()
    if sql.lower().startswith("sql"):
        sql = sql[3:].strip()
    if not sql.lower().startswith("select"):
        raise UnsafeSQLError(f"Only SELECT statements are allowed. Got: {sql[:80]!r}")
    if _DISALLOWED.search(sql):
        raise UnsafeSQLError(f"Query contains a disallowed keyword: {sql[:120]!r}")
    if ";" in sql.strip().rstrip(";"):
        raise UnsafeSQLError("Only a single statement is allowed.")
    return sql


def _llm_configured() -> bool:
    if LLM_PROVIDER == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if LLM_PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def _closest_fallback(question: str) -> tuple[str, str]:
    """Very simple keyword-overlap match against FALLBACK_QUERIES, used both
    when no LLM key is configured and as a last-resort if the LLM call fails."""
    q_words = set(question.lower().split())
    best_label, best_score = None, -1
    for label in FALLBACK_QUERIES:
        score = len(q_words & set(label.split()))
        if score > best_score:
            best_label, best_score = label, score
    return best_label, FALLBACK_QUERIES[best_label]


def generate_sql(question: str) -> tuple[str, bool]:
    """Returns (sql, used_llm). Falls back to a canned query if no key is set."""
    if not _llm_configured():
        _, sql = _closest_fallback(question)
        return sql, False

    schema = get_schema_description()
    system_prompt = SYSTEM_PROMPT.format(schema=schema)

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()  # reads OPENAI_API_KEY from env
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
        )
        raw_sql = resp.choices[0].message.content
    elif LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=300,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        raw_sql = resp.content[0].text
    else:
        raise NotImplementedError(
            f"LLM_PROVIDER='{LLM_PROVIDER}' not wired up yet. Add a branch here, same pattern."
        )

    return _validate_sql(raw_sql), True


def run_sql(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    ensure_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(sql, conn)
    finally:
        conn.close()


def ask(question: str) -> tuple[pd.DataFrame, str, bool]:
    """Full pipeline: question -> validated SQL -> executed -> DataFrame.
    Returns (dataframe, sql, used_llm)."""
    sql, used_llm = generate_sql(question)
    df = run_sql(sql)
    return df, sql, used_llm


if __name__ == "__main__":
    for label, sql in FALLBACK_QUERIES.items():
        print(f"\n--- {label} ---")
        print(run_sql(sql))

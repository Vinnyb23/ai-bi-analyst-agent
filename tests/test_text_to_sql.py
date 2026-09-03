import pytest

from src.text_to_sql import _validate_sql, UnsafeSQLError, ask, FALLBACK_QUERIES, run_sql


def test_validate_sql_accepts_plain_select():
    sql = _validate_sql("SELECT region, SUM(sales) FROM sales GROUP BY region")
    assert sql.lower().startswith("select")


def test_validate_sql_strips_code_fences():
    sql = _validate_sql("```sql\nSELECT * FROM sales\n```")
    assert sql.lower().startswith("select")


@pytest.mark.parametrize(
    "bad_sql",
    [
        "DROP TABLE sales",
        "DELETE FROM sales",
        "SELECT * FROM sales; DROP TABLE sales",
        "UPDATE sales SET sales = 0",
        "not even sql",
    ],
)
def test_validate_sql_rejects_unsafe_statements(bad_sql):
    with pytest.raises(UnsafeSQLError):
        _validate_sql(bad_sql)


def test_ask_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    df, sql, used_llm = ask("total sales by region")
    assert used_llm is False
    assert sql.lower().startswith("select")
    assert len(df) > 0


def test_all_fallback_queries_execute_successfully():
    for label, sql in FALLBACK_QUERIES.items():
        df = run_sql(sql)
        assert df is not None, f"Fallback query '{label}' returned None"

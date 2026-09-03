from src.agents import run_bi_analyst


def test_run_bi_analyst_returns_full_pipeline_result(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    result = run_bi_analyst("What were total sales by region?")

    assert result["used_llm_sql"] is False
    assert result["sql"].lower().startswith("select")
    assert result["row_count"] > 0
    assert len(result["data_summary"]) > 0
    assert result["used_llm_summary"] is False
    assert "report" in result
    assert result["used_llm_report"] is False
    assert "BI Analyst Report" in result["report"]


def test_run_bi_analyst_report_includes_sql_and_summary():
    result = run_bi_analyst("What were total sales by category?")
    assert result["sql"] in result["report"]
    assert result["data_summary"] in result["report"] or len(result["data_summary"]) > 0

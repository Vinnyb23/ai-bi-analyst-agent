from src.rag import retrieve, answer


def test_retrieve_returns_relevant_chunks():
    chunks = retrieve("Grad-CAM explainability heatmap", k=3)
    assert len(chunks) > 0
    for c in chunks:
        assert "text" in c
        assert "source" in c
        assert "start_line" in c


def test_answer_falls_back_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = answer("What is the Phase 2 project about?")
    assert result["used_llm"] is False
    assert len(result["answer"]) > 0
    assert len(result["citations"]) > 0


def test_answer_includes_citation_metadata():
    result = answer("What forecasting model was used in Phase 1?")
    for citation in result["citations"]:
        assert "source" in citation
        assert "start_line" in citation

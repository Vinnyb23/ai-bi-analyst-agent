from src.eval import keyword_score, load_testset, run_eval


def test_keyword_score_full_match():
    assert keyword_score("The model uses ResNet50 and Grad-CAM", ["resnet50", "grad-cam"]) == 1.0


def test_keyword_score_partial_match():
    score = keyword_score("The model uses ResNet50", ["resnet50", "grad-cam"])
    assert score == 0.5


def test_keyword_score_no_expected_keywords_returns_one():
    assert keyword_score("anything", []) == 1.0


def test_load_testset_returns_nonempty_list():
    cases = load_testset()
    assert len(cases) > 0
    for case in cases:
        assert "question" in case
        assert "expected_keywords" in case


def test_run_eval_produces_summary_with_all_rows(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    summary = run_eval()
    cases = load_testset()
    assert summary["n_cases"] == len(cases)
    assert 0.0 <= summary["avg_keyword_score"] <= 1.0
    assert len(summary["rows"]) == len(cases)

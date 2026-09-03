"""
eval.py
-------
Basic evaluation harness for the RAG pipeline (Week 16). Loads a small set
of question/expected-keyword pairs from eval/qa_testset.jsonl, runs each
question through src.rag.answer(), and scores how many expected keywords
show up in the answer text.

This is deliberately simple (keyword-overlap, not a learned metric) but it's
enough to (a) catch regressions when you change chunking/retrieval params,
and (b) let you talk about eval methodology in interviews instead of just
"it looked right when I tried it."

An optional LLM-graded mode (`--llm-grade`) additionally asks the configured
LLM to rate each answer 1-5 for relevance -- skipped automatically (with a
note) if no API key is configured.

Usage:
    python -m src.eval                 # keyword-overlap scoring only
    python -m src.eval --llm-grade      # + LLM-graded relevance (needs a key)
"""

import argparse
import json
import os

from dotenv import load_dotenv

from src.rag import answer

load_dotenv()

TESTSET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "qa_testset.jsonl")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def load_testset(path: str = TESTSET_PATH) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def keyword_score(answer_text: str, expected_keywords: list[str]) -> float:
    """Fraction of expected keywords found (case-insensitive) in the answer text."""
    if not expected_keywords:
        return 1.0
    text_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords)


def _llm_configured() -> bool:
    if LLM_PROVIDER == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if LLM_PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def llm_grade(question: str, answer_text: str) -> float | None:
    """Asks the configured LLM to rate relevance 1-5; returns None if no key set
    or the call fails, so this is purely an optional bonus signal."""
    if not _llm_configured():
        return None
    prompt = (
        f"Question: {question}\nAnswer: {answer_text}\n\n"
        "On a scale of 1-5 (5 = fully and accurately answers the question), "
        "reply with ONLY the number."
    )
    try:
        if LLM_PROVIDER == "openai":
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=LLM_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
            )
            raw = resp.choices[0].message.content
        else:
            import anthropic

            client = anthropic.Anthropic()
            resp = client.messages.create(
                model=LLM_MODEL, max_tokens=10, messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text
        return float("".join(c for c in raw if c.isdigit() or c == "."))
    except Exception:
        return None


def run_eval(testset_path: str = TESTSET_PATH, use_llm_grade: bool = False) -> dict:
    cases = load_testset(testset_path)
    rows = []
    for case in cases:
        result = answer(case["question"])
        score = keyword_score(result["answer"], case["expected_keywords"])
        row = {
            "question": case["question"],
            "keyword_score": round(score, 2),
            "used_llm_answer": result["used_llm"],
            "n_citations": len(result["citations"]),
        }
        if use_llm_grade:
            row["llm_grade"] = llm_grade(case["question"], result["answer"])
        rows.append(row)

    avg_keyword_score = sum(r["keyword_score"] for r in rows) / len(rows) if rows else 0.0
    summary = {"n_cases": len(rows), "avg_keyword_score": round(avg_keyword_score, 3), "rows": rows}
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-grade", action="store_true", help="Also score with an LLM relevance grade (needs an API key)")
    args = parser.parse_args()

    summary = run_eval(use_llm_grade=args.llm_grade)
    for row in summary["rows"]:
        print(f"[{row['keyword_score']:.2f}] {row['question']}")
    print(f"\nAverage keyword-overlap score: {summary['avg_keyword_score']} over {summary['n_cases']} questions")
    if args.llm_grade and not _llm_configured():
        print("(--llm-grade requested but no API key configured -- skipped, see .env.example)")

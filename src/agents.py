"""
agents.py
---------
A 3-agent LangGraph pipeline that mimics a junior BI analyst:

  query_data  -> asks the Phase-1-style sales database a question (src/text_to_sql.py)
  summarize   -> turns the raw query result into a short natural-language summary
  draft_report -> writes a short report combining the summary with any relevant
                  context pulled from the project's own documentation (src/rag.py)

Every node has a deterministic, no-API-key fallback (pandas aggregates for
the summary step, a filled-in template for the report step), so
`run_bi_analyst()` always returns a complete result -- the LLM steps just
upgrade prose quality when a key is configured.

Usage:
    from src.agents import run_bi_analyst
    result = run_bi_analyst("What were total sales by region last quarter?")
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from src.rag import retrieve
from src.text_to_sql import ask as ask_sql

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


class AnalystState(TypedDict, total=False):
    question: str
    sql: str
    used_llm_sql: bool
    data_preview: list
    row_count: int
    data_summary: str
    used_llm_summary: bool
    context_snippets: list
    report: str
    used_llm_report: bool


def _llm_configured() -> bool:
    if LLM_PROVIDER == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if LLM_PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def _call_llm(system_prompt: str, user_message: str, max_tokens: int = 400) -> str:
    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text
    raise NotImplementedError(f"LLM_PROVIDER='{LLM_PROVIDER}' not wired up yet.")


# --- Node 1: query the BI database -----------------------------------------


def node_query_data(state: AnalystState) -> AnalystState:
    df, sql, used_llm_sql = ask_sql(state["question"])
    return {
        "sql": sql,
        "used_llm_sql": used_llm_sql,
        "data_preview": df.head(10).to_dict(orient="records"),
        "row_count": len(df),
    }


# --- Node 2: summarize the result -------------------------------------------


def _deterministic_summary(preview: list, row_count: int) -> str:
    if not preview:
        return "The query returned no rows."
    numeric_cols = [
        k for k, v in preview[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
    ]
    lines = [f"The query returned {row_count} row(s)."]
    for col in numeric_cols:
        values = [row[col] for row in preview if row.get(col) is not None]
        if values:
            lines.append(f"- `{col}`: min={min(values):.2f}, max={max(values):.2f}, avg={sum(values) / len(values):.2f} (first {len(values)} rows)")
    top_row = preview[0]
    lines.append(f"Top row: {top_row}")
    return "\n".join(lines)


def node_summarize(state: AnalystState) -> AnalystState:
    preview, row_count = state["data_preview"], state["row_count"]
    if _llm_configured() and preview:
        try:
            summary = _call_llm(
                "You are a BI analyst. Summarize this SQL query result for a business "
                "stakeholder in 3-4 concise sentences. Call out the biggest number and any "
                "notable pattern.",
                f"Question: {state['question']}\nSQL: {state['sql']}\nRows (sample): {preview}",
                max_tokens=250,
            )
            return {"data_summary": summary, "used_llm_summary": True}
        except Exception:
            pass  # fall through to deterministic summary
    return {"data_summary": _deterministic_summary(preview, row_count), "used_llm_summary": False}


# --- Node 3: draft the report ------------------------------------------------


def node_draft_report(state: AnalystState) -> AnalystState:
    context_chunks = retrieve(state["question"], k=2)
    context_text = "\n".join(f"- {c['text'][:150].strip()}... (source: {c['source']})" for c in context_chunks)

    if _llm_configured():
        try:
            report = _call_llm(
                "You are a BI analyst writing a short report (headline + 2-3 bullet "
                "findings + one recommendation) for a business stakeholder, based on "
                "the SQL result summary and any relevant background context provided.",
                f"Question: {state['question']}\n\nData summary:\n{state['data_summary']}\n\n"
                f"Background context:\n{context_text or 'none'}",
                max_tokens=400,
            )
            return {
                "report": report,
                "used_llm_report": True,
                "context_snippets": [{"source": c["source"], "start_line": c["start_line"]} for c in context_chunks],
            }
        except Exception:
            pass

    report = (
        f"## BI Analyst Report\n\n"
        f"**Question:** {state['question']}\n\n"
        f"**SQL executed:**\n```sql\n{state['sql']}\n```\n\n"
        f"**Findings:**\n{state['data_summary']}\n\n"
        f"**Recommendation:** Review the top result above with the relevant regional/category "
        f"owner; set LLM_PROVIDER + an API key in `.env` for a narrative write-up instead of "
        f"this templated one.\n"
    )
    return {
        "report": report,
        "used_llm_report": False,
        "context_snippets": [{"source": c["source"], "start_line": c["start_line"]} for c in context_chunks],
    }


def build_graph():
    graph = StateGraph(AnalystState)
    graph.add_node("query_data", node_query_data)
    graph.add_node("summarize", node_summarize)
    graph.add_node("draft_report", node_draft_report)
    graph.set_entry_point("query_data")
    graph.add_edge("query_data", "summarize")
    graph.add_edge("summarize", "draft_report")
    graph.add_edge("draft_report", END)
    return graph.compile()


_GRAPH = None


def run_bi_analyst(question: str) -> AnalystState:
    """Runs the full query_data -> summarize -> draft_report pipeline."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    result = _GRAPH.invoke({"question": question})
    return result


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What were total sales by region?"
    result = run_bi_analyst(q)
    print(result["report"])

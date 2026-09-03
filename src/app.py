"""
app.py
------
Streamlit front end tying together the two Phase 3 deliverables:

  Tab 1 - "Ask your portfolio": RAG chat over docs/corpus/ with citations
  Tab 2 - "BI Analyst": the 3-agent LangGraph pipeline (query -> summarize -> report)

Usage:
    streamlit run src/app.py
"""

import os
import sys

# Same shim as Phase 1/2: `streamlit run src/app.py` executes with the
# script's own directory as sys.path[0], not the project root, so
# `from src.xxx import ...` fails without this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from src.agents import run_bi_analyst
from src.rag import answer

st.set_page_config(page_title="AI BI Analyst Agent", page_icon="🧭", layout="wide")

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
_llm_configured = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))

with st.sidebar:
    st.header("About")
    st.write(f"**LLM provider:** `{LLM_PROVIDER}`")
    st.write(f"**API key configured:** {'yes' if _llm_configured else 'no (fallback mode)'}")
    st.markdown("---")
    st.write(
        "Phase 3 project: RAG chatbot over this program's own project docs, plus a "
        "3-agent LangGraph pipeline that queries a BI database, summarizes the "
        "result, and drafts a short report."
    )
    if not _llm_configured:
        st.info(
            "No OPENAI_API_KEY / ANTHROPIC_API_KEY set -- running in fallback mode "
            "(raw retrieved passages instead of synthesized prose, templated report "
            "instead of an LLM-written one). Set a key in `.env` for the full experience."
        )

st.title("🧭 AI BI Analyst Agent")

tab_rag, tab_analyst = st.tabs(["💬 Ask your portfolio (RAG)", "📊 BI Analyst (multi-agent)"])

with tab_rag:
    st.subheader("Ask a question about your own Phase 1-3 project docs")
    st.caption("Retrieval-augmented answers with citations back to the exact file + line.")

    example_qs = [
        "What critical bug did the vision classifier project have and how was it fixed?",
        "What forecasting model does the Phase 1 project use?",
        "What vector database and embedding model does this RAG pipeline use?",
    ]
    question = st.text_input("Your question", placeholder=example_qs[0])
    cols = st.columns(len(example_qs))
    for col, eq in zip(cols, example_qs):
        if col.button(eq, use_container_width=True):
            question = eq

    if question:
        with st.spinner("Retrieving + answering..."):
            result = answer(question)
        st.markdown(f"**Answer** {'(LLM-synthesized)' if result['used_llm'] else '(fallback: raw retrieved passages)'}")
        st.write(result["answer"])
        if result["citations"]:
            st.markdown("**Citations**")
            for c in result["citations"]:
                st.caption(f"{c['source']} — line {c['start_line']}")

with tab_analyst:
    st.subheader("Ask the multi-agent BI analyst")
    st.caption("query_data → summarize → draft_report, run with LangGraph over a synthetic sales DB.")

    example_bi_qs = [
        "What were total sales by region?",
        "What were total sales by category?",
        "Which sub-categories are most profitable?",
    ]
    bi_question = st.text_input("Your BI question", placeholder=example_bi_qs[0], key="bi_question")
    bi_cols = st.columns(len(example_bi_qs))
    for col, eq in zip(bi_cols, example_bi_qs):
        if col.button(eq, use_container_width=True, key=f"bi_{eq}"):
            bi_question = eq

    if bi_question:
        with st.spinner("Running query_data -> summarize -> draft_report..."):
            result = run_bi_analyst(bi_question)

        st.markdown("**Generated SQL**")
        st.code(result["sql"], language="sql")

        st.markdown(f"**Data summary** {'(LLM)' if result['used_llm_summary'] else '(deterministic)'}")
        st.write(result["data_summary"])

        if result.get("data_preview"):
            st.markdown("**Result preview**")
            st.dataframe(result["data_preview"])

        st.markdown(f"**Report** {'(LLM-written)' if result['used_llm_report'] else '(templated)'}")
        st.write(result["report"])

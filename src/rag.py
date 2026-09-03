"""
rag.py
------
Retrieval-augmented answering over the docs/corpus/ portfolio documents.

retrieve(question, k)  -> top-k chunks with source + line citations
answer(question, k)    -> {answer, citations, used_llm}

Zero-key fallback: if no LLM_PROVIDER key is configured, `answer()` skips
generation and returns the retrieved chunks themselves (concatenated, each
tagged with its citation) instead of an LLM-written paragraph -- so the RAG
retrieval + citation behavior (the actual point of Weeks 13-14) is fully
demonstrable without any API key, only the prose synthesis step needs one.

Usage:
    from src.rag import answer
    result = answer("What bug did the vision classifier project have?")
"""

import os

from dotenv import load_dotenv

from src.ingest import build_or_load_vectorstore, embed_texts

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant answering questions about the user's own \
AI/ML learning-program project documentation. Answer ONLY using the provided context chunks. \
If the context doesn't contain the answer, say so plainly -- never make things up. After each \
claim, cite the source in the form (source: <file>, line <n>) using the citations given.

Context:
{context}
"""


def _llm_configured() -> bool:
    if LLM_PROVIDER == "openai":
        return bool(os.getenv("OPENAI_API_KEY"))
    if LLM_PROVIDER == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    return False


def retrieve(question: str, k: int = 4) -> list[dict]:
    """Returns the top-k chunks as {text, source, start_line, distance}."""
    collection = build_or_load_vectorstore()
    if collection.count() == 0:
        return []
    query_embedding = embed_texts([question])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=min(k, collection.count()))

    chunks = []
    for text, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        chunks.append(
            {
                "text": text,
                "source": meta["source"],
                "start_line": meta["start_line"],
                "distance": dist,
            }
        )
    return chunks


def _format_citation(chunk: dict) -> str:
    return f"(source: {chunk['source']}, line {chunk['start_line']})"


def _escape_markdown_lead(text: str) -> str:
    """Quoted document text often starts with '#', '-', '*', or '>' (e.g. a
    source file's own heading or bullet). Rendered verbatim in a markdown
    viewer, a leading '#' turns the whole snippet into a giant heading, and
    stripping the marker outright can leave an unpaired trailing '**' from
    bold text. Backslash-escape the leading run instead, so it displays as
    plain literal characters with no dangling formatting markers."""
    import re

    return re.sub(r"^(\s*)([#\-*>]+)", lambda m: m.group(1) + "\\" + "\\".join(m.group(2)), text)


def _fallback_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "No matching context was found in the indexed documents."
    parts = []
    for c in chunks:
        snippet = _escape_markdown_lead(c["text"].strip().replace("\n", " "))
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + "..."
        parts.append(f"{snippet} {_format_citation(c)}")
    return (
        "No LLM_PROVIDER API key is configured, so here are the most relevant "
        "retrieved passages directly (set OPENAI_API_KEY or ANTHROPIC_API_KEY "
        "in .env for a synthesized answer):\n\n" + "\n\n".join(parts)
    )


def _llm_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"{_format_citation(c)}\n{c['text']}" for c in chunks)
    system_prompt = ANSWER_SYSTEM_PROMPT.format(context=context)

    if LLM_PROVIDER == "openai":
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content
    elif LLM_PROVIDER == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": question}],
        )
        return resp.content[0].text
    else:
        raise NotImplementedError(f"LLM_PROVIDER='{LLM_PROVIDER}' not wired up yet.")


def answer(question: str, k: int = 4) -> dict:
    """Full RAG pipeline: retrieve -> (LLM synthesis | raw-chunk fallback)."""
    chunks = retrieve(question, k=k)
    used_llm = _llm_configured() and bool(chunks)

    if used_llm:
        try:
            text = _llm_answer(question, chunks)
        except Exception as exc:  # network/quota/etc. -- degrade gracefully, don't crash the demo
            used_llm = False
            text = _fallback_answer(chunks) + f"\n\n(LLM call failed: {exc})"
    else:
        text = _fallback_answer(chunks)

    return {
        "answer": text,
        "citations": [{"source": c["source"], "start_line": c["start_line"]} for c in chunks],
        "used_llm": used_llm,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is this program's Phase 2 project about?"
    result = answer(q)
    print(f"Q: {q}\n")
    print(result["answer"])
    print("\nCitations:", result["citations"])

# Learning Log — Phase 3: AI BI Analyst Agent

Running notes as I work through Weeks 13-18 of the program. Format follows the same pattern as Phase 1 and Phase 2's logs: what I built, what I learned, what tripped me up.

## Week 13-14: RAG pipeline

**What I built:** A document-ingestion pipeline (`src/ingest.py`) that chunks markdown files with `RecursiveCharacterTextSplitter`, embeds them locally with `sentence-transformers` (`all-MiniLM-L6-v2`, no API key), and stores everything in a persistent Chroma collection. `src/rag.py` retrieves the top-k chunks for a question and either synthesizes a cited answer with an LLM or, with no key configured, returns the raw retrieved passages with citations.

**Corpus choice:** Instead of pulling in an external dataset, I pointed the RAG pipeline at my own Phase 1/2 READMEs, learning logs, and the program plan itself. It's a more meaningful demo (asking the bot about my own past mistakes and decisions) and it meant zero download/credential friction to get started.

**What I learned:**
- `RecursiveCharacterTextSplitter`'s separator list matters a lot for chunk quality on markdown — using `["\n## ", "\n### ", "\n\n", "\n", " ", ""]` keeps section headers attached to their content instead of splitting mid-heading.
- Chroma's `PersistentClient` + `get_or_create_collection` is enough for local persistence — no server process needed for a project this size.
- Designing the fallback path *first* (raw chunks + citations, no LLM) actually made the retrieval logic better, because I had to get citation metadata (source file + line number) right from the start rather than treating it as an afterthought once LLM synthesis was working.

## Week 15: multi-agent BI analyst

**What I built:** A 3-node LangGraph pipeline: `query_data` (reuses Phase 1's `text_to_sql.py` to turn a question into SQL and run it), `summarize` (turns the raw DataFrame into plain-English findings), `draft_report` (combines the summary with relevant RAG context into a short report). Each node has a deterministic fallback so the whole graph runs and produces a complete, readable result with zero API keys configured.

**What I learned:**
- LangGraph's `StateGraph` + `TypedDict` state is a clean way to express a linear pipeline, but the real value is that each node is independently testable — I could unit-test `node_summarize` on a hand-built fake state without running the SQL step first.
- Keeping the sales database standalone (this repo's own `data_prep.py`, not imported from `bi-ai-assistant`) was the right call for portability, even though it meant a small amount of duplicated code — matches the same lesson from Phase 2 about self-contained repos.

## Week 16: evaluation harness

**What I built:** `eval/qa_testset.jsonl` with 8 hand-written questions about the project's own documentation, each tagged with expected keywords, scored via simple keyword-overlap in `src/eval.py`. Added an optional `--llm-grade` mode for relevance scoring if a key is configured.

**What I learned:**
- Keyword-overlap scoring is crude but genuinely useful as a regression check — I can change chunk size or retrieval `k` and immediately see if scores drop, without needing an LLM judge or a labeled gold-answer set.
- The zero-key baseline score (0.48 average keyword overlap) is a legitimate number to report, not a placeholder — it's the honest floor for what raw retrieval-without-synthesis gets you, and a fair before/after comparison point once an LLM key is added.

## Week 17: Streamlit app

**What I built:** A two-tab Streamlit app — RAG chat with citations, and the BI analyst pipeline with SQL/summary/report all shown so the pipeline's steps are visible rather than just the final output.

**What I learned:**
- Same `sys.path.insert(...)` shim from Phase 1/2 was needed again for `streamlit run src/app.py` — worth just keeping as a standard first two lines in any future `app.py`.

## Week 18: deployment (next)

- [ ] Deploy to Hugging Face Spaces (`vinnyb23/ai-bi-analyst-agent`)
- [ ] Confirm the app runs correctly in fallback mode with no API key on the Space
- [ ] Add the live demo link to README.md

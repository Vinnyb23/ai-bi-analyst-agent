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

## Week 18: deployment

- [x] Deploy to Hugging Face Spaces (`vinnyb23/ai-bi-analyst-agent`)
- [x] Confirm the app runs correctly in fallback mode with no API key on the Space
- [ ] Add the live demo link to README.md

## Week 19: CI/CD (Phase 4 practice run)

**What I built:** A GitHub Actions workflow (`.github/workflows/ci.yml`) that lints with [ruff](https://docs.astral.sh/ruff/) and runs the full `pytest` suite on every push/PR to `main`. Split lint tooling into its own `requirements-dev.txt` rather than bloating the app's production `requirements.txt`.

**What I learned:**
- Ruff's *default* rule set on a fresh install turned out to be more opinionated than expected — it flagged every `except Exception:` fallback block (the exact pattern this repo relies on for graceful LLM degradation) as an error under `BLE001`/`S110`. Explicitly pinning `select = ["E", "F", "I"]` in `pyproject.toml` keeps CI focused on real bugs, style, and import order instead of fighting an intentional design choice — a good reminder to always pin an explicit lint config rather than trusting a tool's defaults.
- CI needs to target the *same* Python version the Dockerfile/Space use (3.14, because of the `numpy==2.5.2` wheel constraint from Week 17-18), not just whatever's convenient — caught this before it became a mismatched, confusing CI failure.
- Ran the exact lint + test commands locally first, matching what the workflow runs, so the first real CI run isn't also the first real test of the commands themselves.

**Next (Phase 4, Weeks 20-24):** MLflow experiment tracking/model registry, drift monitoring with Evidently AI, and the unified capstone app tying Phases 1-3 together.

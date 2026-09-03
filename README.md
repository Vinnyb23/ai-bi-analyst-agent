---
title: AI BI Analyst Agent
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: streamlit
sdk_version: "1.63.0"
python_version: "3.14"
app_file: src/app.py
pinned: false
---

# AI BI Analyst Agent

**Phase 3 project** of a 6-month self-directed AI/ML continuing-education program (following the UT Austin PGP-AI certificate). This phase moves from single-model ML (Phases 1-2) into generative AI and agentic workflows: a RAG chatbot over the program's own project documentation, and a 3-node LangGraph pipeline that plays "junior BI analyst" — it queries a database, summarizes what it found, and drafts a written report.

> Live demo: _add your Hugging Face Spaces link here after deploying (Week 18)_

![Python](https://img.shields.io/badge/python-3.14-blue)
![LangGraph](https://img.shields.io/badge/agents-LangGraph-1C3C3C)
![Chroma](https://img.shields.io/badge/vector%20store-Chroma-FF6F00)
![sentence--transformers](https://img.shields.io/badge/embeddings-sentence--transformers-yellow)

## What it does

- **RAG chat over your own portfolio** — the document corpus is this program's own Phase 1 and Phase 2 READMEs, learning logs, and the program plan, so asking "what bug did the vision classifier have?" gets a cited answer pulled straight from your own past work.
- **Cited retrieval** — every chunk is tagged with its source file and starting line number, so answers can always point back to exactly where a claim came from.
- **3-agent BI analyst pipeline (LangGraph)** — `query_data` turns a question into SQL and runs it against a synthetic sales database (reusing Phase 1's text-to-SQL layer), `summarize` turns the raw result into plain-English findings, `draft_report` writes a short report combining the summary with any relevant background context from the RAG corpus.
- **Evaluation harness** — a small labeled test set (`eval/qa_testset.jsonl`) scored by keyword-overlap, with an optional LLM-graded relevance mode if a key is configured.
- **Zero-setup by default** — no API key, external database, or cloud vector DB account required to run every script and test. See "LLM configuration" below.

## LLM configuration

Every LLM-touching step (SQL generation, RAG answer synthesis, data summarization, report drafting) follows the same provider-agnostic pattern used in Phase 1 and Phase 2: set `LLM_PROVIDER` (`openai` or `anthropic`) and the matching API key in `.env`, copied from `.env.example`.

**With no API key configured, every feature still works end to end**, just with simpler output instead of LLM-generated prose:

| Step | With API key | Without API key (fallback) |
|---|---|---|
| Text-to-SQL | LLM writes the SQL | Keyword-matched against 5 canned queries |
| RAG answer | LLM synthesizes a cited paragraph | Raw top-k retrieved passages, each tagged with its citation |
| Data summary | LLM writes 3-4 sentences | Pandas min/max/avg + top row, as text |
| Report draft | LLM writes headline + findings + recommendation | Templated report using the SQL, summary, and retrieved context |
| Eval scoring | + optional LLM relevance grade (1-5) | Keyword-overlap score only |

Embeddings are always local (`sentence-transformers`, no API key ever required for the RAG retrieval step itself — only the optional answer-synthesis step needs a key).

## Architecture

```
docs/corpus/*.md ──> src/ingest.py ──> data/chroma/ (Chroma vector store)
                                            │
                                            v
                                     src/rag.py ──> answer(question) ──┐
                                                                        │
data/bi_analyst.db (synthetic) <── src/data_prep.py                    │
        │                                                              │
        v                                                              v
src/text_to_sql.py ──> src/agents.py (LangGraph: query_data ──────────►
                          -> summarize -> draft_report)                │
        │                                                              │
        └──────────────────────> src/app.py (Streamlit: 2 tabs) <──────┘
                                       │
                                  src/eval.py (scores src/rag.py answers)
```

## Project structure

```
ai-bi-analyst-agent/
├── docs/corpus/            # RAG document corpus (this program's own READMEs/logs/plan)
├── data/                   # synthetic sales DB + Chroma vector store (gitignored, regenerated)
├── eval/qa_testset.jsonl   # labeled Q&A test set for src/eval.py
├── notebooks/01_exploration.ipynb
├── src/
│   ├── data_prep.py        # synthetic sales DB generator (own copy, portable)
│   ├── text_to_sql.py      # provider-agnostic NL -> SQL + safety guardrail
│   ├── ingest.py           # chunk + embed docs/corpus/ into Chroma
│   ├── rag.py               # retrieve + cite + (LLM | fallback) answer synthesis
│   ├── agents.py            # 3-node LangGraph BI analyst pipeline
│   ├── eval.py               # keyword-overlap (+ optional LLM-graded) scoring
│   └── app.py                # Streamlit UI: RAG chat + BI analyst tabs
├── tests/                   # pytest suite, all pass in zero-key fallback mode
├── requirements.txt
├── Dockerfile
├── .env.example
└── .gitignore
```

## Setup

> **Note:** this repo pins `numpy==2.5.2`, which as of this writing only ships a Python 3.14 wheel — use Python 3.14 locally (or the provided Dockerfile/Space config, both already pinned to 3.14) rather than 3.11/3.12.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

cp .env.example .env   # optionally add an API key; works fine with it left blank

python src/data_prep.py     # builds data/bi_analyst.db
python -m src.ingest         # builds data/chroma/ from docs/corpus/

streamlit run src/app.py
```

## Running the evaluation

```bash
python -m src.eval                 # keyword-overlap scoring
python -m src.eval --llm-grade      # + LLM-graded relevance (needs an API key)
```

## Running tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker build -t ai-bi-analyst-agent .
docker run -p 8501:8501 --env-file .env ai-bi-analyst-agent
```

## Roadmap (Weeks 13-18)

- [x] Week 13-14: document corpus + chunking + Chroma ingestion + cited retrieval
- [x] Week 14: RAG answer synthesis with zero-key fallback
- [x] Week 15: LangGraph 3-agent BI analyst pipeline (query -> summarize -> report)
- [x] Week 16: evaluation harness (keyword-overlap + optional LLM grading)
- [x] Week 17: Streamlit app tying both features together
- [ ] Week 18: deploy to Hugging Face Spaces, add live demo link above

## Part of a larger program

This is Phase 3 of a self-directed 6-month AI/ML continuing-education program. See the sibling repos for earlier phases:

- **Phase 1:** `bi-ai-assistant` — text-to-SQL BI copilot with forecasting
- **Phase 2:** `explainable-vision-classifier` — Grad-CAM explainable defect classifier (VGG-16 vs. fine-tuned ResNet50)
- **Phase 3 (this repo):** `ai-bi-analyst-agent` — RAG chatbot + multi-agent BI analyst

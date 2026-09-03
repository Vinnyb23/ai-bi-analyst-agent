# From UT Austin PGP-AI to AI/ML Engineer
### A 6-Month Self-Directed Continuing Education Program

**Built for:** Benson — Business Intelligence Developer, 15 years in data analytics, graduate of UT Austin's Post Graduate AI/ML Program (CNNs, VGG-16 transfer learning, Random Forest/XGBoost, Flask, Streamlit, Docker)

**Pace:** 3–5 hrs/week for 24 weeks (~6 months)
**Focus areas (all four, rotated in phases):** Generative AI & LLM agents · Computer Vision · MLOps & Deployment · BI+AI Fusion
**Home base:** GitHub (you already have an account) + optional live demos on Hugging Face Spaces

---

## Why this structure

Your UT Austin coursework already covered the hard part — the math and modeling fundamentals (CNNs, transfer learning, ensemble methods, imbalanced classification like EasyVisa and ReneWind, and basic deployment like SuperKart). What's usually missing after a bootcamp-style program is: **build habits that turn coursework into a public, employer-visible track record.**

Two things stand out from the 2026 hiring landscape worth aligning to from day one:
- Retrieval-Augmented Generation (RAG) and agentic workflows (tool-use, multi-step reasoning loops) are now considered core AI-engineering skills, not niche add-ons ([AI for Anything, 2026](https://www.aiforanything.io/blog/ai-skills-to-learn-2026), [MC Taba, 2026](https://www.mctaba.com/learn/software-ai-engineering/ai-skills-needed)).
- A credible data science/ML portfolio in 2026 is expected to have 3–5 *runnable* GitHub repos with real READMEs, plus 1–2 live deployed demos — not just notebooks ([Data Scientist Portfolio Guide, 2026](https://www.naildd.com/blog/data-scientist-portfolio-guide)).
- MLflow remains the de facto standard for experiment tracking and model registry across almost every 2026 MLOps stack comparison, making it the highest-leverage tool to learn first for productionizing models ([MLflow, 2026](https://mlflow.org/articles/mlops-pipeline-automation-best-practices-in-2026/), [Dupple, 2026](https://dupple.com/learn/best-mlops-tools)).

So this program is designed around **one flagship, portfolio-worthy project per phase**, each shipped to GitHub with a professional README, and increasingly tied together so your capstone in Month 6 unifies all four skill areas — mirroring the SaaS product you've mentioned wanting to build.

---

## Program at a glance

| Phase | Weeks | Focus | Flagship project | Builds on |
|---|---|---|---|---|
| 1 | 1–6 | BI + AI Fusion | Natural-language BI assistant (text-to-SQL + forecasting) | Your 15 yrs of BI/SQL experience |
| 2 | 7–12 | Computer Vision | Explainable defect/image classifier with Grad-CAM | Your VGG-16/transfer-learning work |
| 3 | 13–18 | Generative AI & Agents | RAG chatbot + multi-agent report-writing pipeline | New — highest-demand 2026 skill |
| 4 | 19–24 | MLOps & Deployment (Capstone) | Unify Phases 1–3 into one deployed, monitored SaaS-style app | Docker/Flask/Streamlit from UT Austin |

Each phase = 6 weeks × 3–5 hrs = roughly 24–30 hours, enough for one well-documented project rather than several rushed ones. Depth over breadth is the goal — quality repos beat quantity.

---

## Phase 1 (Weeks 1–6): BI + AI Fusion

**Goal:** Combine your strongest existing skill (BI/SQL/reporting) with ML so you have an immediate "unfair advantage" project that's hard for a pure-CS bootcamp grad to replicate.

**Learning objectives**
- Time-series forecasting (Prophet or statsmodels/XGBoost) applied to a BI-style KPI dataset
- Text-to-SQL with an LLM (turn plain-English questions into SQL against a real database)
- Packaging BI outputs as an interactive Streamlit dashboard

**Weekly breakdown**
- Week 1: Pick a public retail/sales dataset (e.g., Kaggle "Superstore" or reuse a SuperKart-style dataset). Set up repo, virtual env, load into a local Postgres/SQLite DB.
- Week 2: Build a forecasting model (XGBoost or Prophet) for sales/demand; log experiments in MLflow (introduce it here so it's a habit for every later project).
- Week 3: Add a natural-language-to-SQL layer using an LLM API (OpenAI, Anthropic, or a local model) so users can ask "what were Q2 sales in the Southeast?" and get a live query + chart.
- Week 4: Wrap it in a Streamlit dashboard combining the forecast chart + NL query box.
- Week 5: Write tests, add a `requirements.txt`/`environment.yml`, Dockerize it.
- Week 6: Polish README (problem, approach, screenshots, how to run), push to GitHub, deploy a live demo on Hugging Face Spaces or Streamlit Community Cloud.

**Deliverable:** `bi-ai-assistant` repo with working text-to-SQL + forecasting dashboard, live demo link.

---

## Phase 2 (Weeks 7–12): Computer Vision, leveled up

**Goal:** Move beyond the VGG-16 transfer-learning exercises from UT Austin into a project with real explainability and a more modern architecture — this is what differentiates a coursework repo from a portfolio-grade one.

**Learning objectives**
- Fine-tuning a modern CV backbone (ResNet/EfficientNet, or a Vision Transformer) vs. VGG-16
- Model explainability (Grad-CAM / SHAP for images)
- Serving a vision model behind a simple API

**Weekly breakdown**
- Week 7: Choose a dataset with real-world framing (manufacturing defect detection, plant disease, or X-ray/skin-lesion classification — pick something you can talk about in an interview). Baseline with your existing VGG-16 approach for comparison.
- Week 8: Fine-tune a stronger backbone (ResNet50/EfficientNet or a small ViT) via transfer learning; log both runs in MLflow so you can show a *before/after model comparison* — a strong storytelling angle for your README.
- Week 9: Add Grad-CAM heatmaps so predictions are explainable (huge credibility booster, especially for regulated domains like defect/medical imaging).
- Week 10: Wrap the model in a Flask API (`/predict` endpoint) — reusing the Flask skills from UT Austin.
- Week 11: Build a lightweight Streamlit or plain HTML front end that uploads an image and shows prediction + heatmap.
- Week 12: Dockerize, write the README with the model comparison table and example heatmap images, push to GitHub, deploy demo.

**Deliverable:** `explainable-vision-classifier` repo with model comparison (VGG-16 baseline vs. fine-tuned backbone), Grad-CAM visuals, deployed demo.

---

## Phase 3 (Weeks 13–18): Generative AI & Agentic Workflows

**Goal:** This is the newest and highest-leverage skill area in 2026 hiring — RAG and agentic tool-use are now treated as baseline AI-engineering skills, not extras ([AI for Anything, 2026](https://www.aiforanything.io/blog/ai-skills-to-learn-2026)). This phase is deliberately the most hands-on-new-territory.

**Learning objectives**
- Retrieval-Augmented Generation (RAG): chunking, embeddings, vector search
- Multi-step agent design (tool-use, reasoning loops) with a framework like LangGraph or CrewAI
- Evaluating LLM output quality (not just "it works," but measuring it)

**Weekly breakdown**
- Week 13: Build a RAG chatbot over a document set you actually care about (your own UT Austin coursework notes/slides, or public BI/finance docs). Use a vector DB (Chroma, FAISS, or Pinecone free tier) + embeddings.
- Week 14: Add source citation to responses (show which chunk/doc backed each answer) — this is what separates a toy RAG demo from a credible one.
- Week 15: Design a simple multi-agent pipeline: one agent queries your Phase 1 BI database, another summarizes findings, another drafts a written report — essentially an "AI BI analyst." Use LangGraph or CrewAI.
- Week 16: Add basic evaluation — a small test set of Q&A pairs and a scoring script (even simple exact-match/LLM-graded scoring counts) so you can talk about eval rigor in interviews.
- Week 17: Wrap the agent pipeline in a simple chat UI (Streamlit or a small React/Flask front end).
- Week 18: README with architecture diagram (agents + tools + data flow), push to GitHub, deploy demo (note: keep API keys out of the repo — use `.env` + secrets in the hosting platform).

**Deliverable:** `ai-bi-analyst-agent` repo — RAG + multi-agent pipeline that autonomously queries, analyzes, and reports on data, with citations and a basic eval harness.

---

## Phase 4 (Weeks 19–24): MLOps, Deployment & Capstone

**Goal:** Tie everything together into one production-grade, monitored, CI/CD-deployed application — directly feeding your interest in SaaS product development.

**Learning objectives**
- CI/CD for ML (GitHub Actions running tests + linting on every push)
- Experiment tracking and model registry with MLflow (you've been logging runs since Phase 1 — now formalize it)
- Basic production monitoring / drift detection (Evidently AI is a good lightweight choice)
- Cloud deployment beyond localhost (Render, Fly.io, or AWS free tier, plus Hugging Face Spaces for demos)

**Weekly breakdown**
- Week 19: Set up a GitHub Actions workflow (lint + test) on one of your Phase 1–3 repos as practice.
- Week 20: Stand up a proper MLflow tracking server (even a local/free-tier one) and migrate the experiment logs from Phases 1–2 into it as a model registry.
- Week 21: Add drift/monitoring instrumentation with Evidently AI to your Phase 2 vision model or Phase 1 forecasting model.
- Week 22–23: Build the **capstone**: a single unified app — e.g., a "BI Copilot" SaaS demo where a user uploads data, gets a forecast (Phase 1), can classify/inspect images if relevant (Phase 2), and can chat with an agent that explains the results and drafts a report (Phase 3) — all containerized with Docker Compose and deployed with a CI/CD pipeline.
- Week 24: Final polish across ALL repos: consistent README template, pinned repos on your GitHub profile, and a short top-level GitHub profile README summarizing the whole program as a narrative ("6-month self-directed AI engineering program following my UT Austin PGP-AI certificate").

**Deliverable:** `bi-copilot-capstone` repo (or similar name) — the unifying SaaS-style demo, deployed and monitored, plus a fully curated GitHub profile.

---

## Your GitHub portfolio structure

Apply this same template to every repo so reviewers immediately trust the work ([Data Scientist Portfolio Guide, 2026](https://www.naildd.com/blog/data-scientist-portfolio-guide); [Python DS/ML Roadmap Portfolio Guide](https://djordjeperovic.github.io/python-ds-ml-roadmap/portfolio/)):

```
repo-name/
├── README.md          <- problem, approach, results, screenshots, how to run, live demo link
├── requirements.txt / environment.yml
├── Dockerfile
├── data/               <- small samples only, or a script to fetch data (never commit large/raw datasets)
├── notebooks/          <- exploration & modeling notebooks
├── src/                <- production-style code (data prep, model, API, app)
├── tests/              <- at least a few automated tests
├── .github/workflows/  <- CI (lint/test) once you reach Phase 4
└── mlruns/ or a note pointing to your MLflow tracking server
```

**README must-haves:** one-paragraph problem statement, a results table or chart, at least one screenshot/GIF, "how to run locally," and a live demo link where applicable.

**GitHub profile-level:**
- Pin your 4 flagship repos (one per phase) plus the capstone.
- Add a top-level profile README (a repo named exactly your username) with a short narrative: your BI background → UT Austin certificate → this 6-month program → what you're aiming for next (SaaS/AI engineering role).
- Keep API keys/secrets out of every repo — use `.env` files (gitignored) and platform-level secrets when deploying.

---

## Tools you'll pick up across the program

| Category | Tool | Where it appears |
|---|---|---|
| Experiment tracking / model registry | MLflow | Phases 1, 2, 4 (industry-standard default in 2026 — [MLflow](https://mlflow.org/articles/mlops-pipeline-automation-best-practices-in-2026/), [Dupple](https://dupple.com/learn/best-mlops-tools)) |
| Forecasting | XGBoost / Prophet | Phase 1 |
| Vector DB | Chroma / FAISS | Phase 3 |
| Agent framework | LangGraph or CrewAI | Phase 3 |
| Monitoring / drift | Evidently AI | Phase 4 |
| CI/CD | GitHub Actions | Phase 4 |
| Deployment | Docker, Hugging Face Spaces, Render/Fly.io | All phases |
| Front end | Streamlit (fast) or light Flask/HTML | All phases |

---

## Suggested weekly rhythm (3–5 hrs/week)

- ~1 hr: read/watch conceptual material for that week's topic
- ~2–3 hrs: hands-on build in Colab/local, commit progress to a feature branch
- ~1 hr: write/update README, reflect on what you learned (a running "learning log" markdown file in each repo is a nice touch reviewers appreciate)

## Staying accountable

- Commit something every week, even small — GitHub's contribution graph is a lightweight but real signal to reviewers.
- At the end of each phase, write a short LinkedIn post or blog note summarizing the project — this compounds your visibility while you learn.
- Revisit your EasyVisa, ReneWind, and SuperKart notebooks from UT Austin at the start of Phases 1 and 2 respectively — they're strong baselines to explicitly reference and improve on in your new repos ("v2 of my UT Austin capstone, now deployed and explainable").

---

*Program compiled August 26, 2026. Tool/landscape references current as of that date — re-check tool rankings roughly every 6 months since the AI tooling ecosystem moves fast.*

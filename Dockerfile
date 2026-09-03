# Pinned to 3.14 to match the exact Python version the requirements.txt
# versions were resolved and tested against -- numpy==2.5.2 (and possibly
# other pins) only ship a cp314 wheel as of this writing, so python:3.11-slim
# would fail here with "no matching distribution found".
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

# Seed the sample sales DB and build the RAG vector store at build time so
# the container works out of the box with no setup step.
RUN python src/data_prep.py && python -m src.ingest

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]

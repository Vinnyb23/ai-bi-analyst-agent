"""
ingest.py
---------
RAG ingestion pipeline: loads the markdown corpus in docs/corpus/ (this
project's own portfolio -- Phase 1 and Phase 2 READMEs/learning logs, plus
the 6-month program plan), chunks it, embeds chunks locally with
sentence-transformers (no API key needed), and stores everything in a
persistent Chroma vector store at data/chroma/.

Using your own project docs as the corpus means Week 13's "document set you
actually care about" doubles as a live demo: ask this repo questions about
your other two repos and get cited answers back.

Usage:
    python -m src.ingest            # build/rebuild the vector store
"""

import glob
import os

from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "corpus")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma")
COLLECTION_NAME = "portfolio_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

_embedder = None  # lazy-loaded singleton, avoids reloading the model per call


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL_NAME)
    return _embedder


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    return model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True).tolist()


def load_corpus_documents(corpus_dir: str = CORPUS_DIR) -> list[dict]:
    """Reads every .md file in corpus_dir into {source, text}."""
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"source": os.path.basename(path), "text": text})
    return docs


def chunk_documents(
    docs: list[dict], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Splits each document into overlapping chunks and tags each chunk with
    its source filename, index, and the 1-based line number it starts on
    (for human-readable citations)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
    )
    chunks = []
    for doc in docs:
        text = doc["text"]
        pieces = splitter.split_text(text)
        cursor = 0
        for i, piece in enumerate(pieces):
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            start_line = text.count("\n", 0, start) + 1
            cursor = max(start, cursor)
            chunks.append(
                {
                    "id": f"{doc['source']}::chunk{i}",
                    "source": doc["source"],
                    "chunk_index": i,
                    "start_line": start_line,
                    "text": piece,
                }
            )
    return chunks


def build_or_load_vectorstore(
    persist_dir: str = CHROMA_DIR, corpus_dir: str = CORPUS_DIR, force_rebuild: bool = False
):
    """Returns a Chroma collection populated with the corpus. Rebuilds only
    if the collection is empty or force_rebuild is True -- re-embedding on
    every call would be slow and pointless once the corpus is stable."""
    os.makedirs(persist_dir, exist_ok=True)
    client = PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if force_rebuild:
        client.delete_collection(COLLECTION_NAME)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        docs = load_corpus_documents(corpus_dir)
        chunks = chunk_documents(docs)
        if not chunks:
            return collection
        embeddings = embed_texts([c["text"] for c in chunks])
        collection.add(
            ids=[c["id"] for c in chunks],
            embeddings=embeddings,
            documents=[c["text"] for c in chunks],
            metadatas=[
                {"source": c["source"], "chunk_index": c["chunk_index"], "start_line": c["start_line"]}
                for c in chunks
            ],
        )
    return collection


if __name__ == "__main__":
    collection = build_or_load_vectorstore(force_rebuild=True)
    print(f"Indexed {collection.count()} chunks from {CORPUS_DIR} into {CHROMA_DIR}")

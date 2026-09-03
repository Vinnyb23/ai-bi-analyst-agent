from src.ingest import chunk_documents, load_corpus_documents, embed_texts


def test_load_corpus_documents_finds_markdown_files():
    docs = load_corpus_documents()
    assert len(docs) > 0
    for doc in docs:
        assert doc["source"].endswith(".md")
        assert len(doc["text"]) > 0


def test_chunk_documents_produces_chunks_with_metadata():
    docs = [{"source": "fake.md", "text": "A" * 2000}]
    chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        assert c["source"] == "fake.md"
        assert "start_line" in c
        assert "id" in c


def test_chunk_documents_respects_small_chunk_size():
    docs = [{"source": "fake.md", "text": "word " * 400}]
    chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)
    assert all(len(c["text"]) <= 250 for c in chunks)  # small overshoot allowed by splitter


def test_embed_texts_returns_vectors_of_consistent_length():
    vectors = embed_texts(["hello world", "another sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0

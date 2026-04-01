from pathlib import Path

from langchain_core.documents import Document

import rag_arena.retrieval as retrieval_module
from rag_arena.retrieval import (
    build_bm25_retriever,
    build_faiss_retriever,
    build_hybrid_retriever,
    build_retriever,
)


def _sample_documents():
    return [
        Document(
            page_content="Paris is the capital city of France.",
            metadata={"doc_id": "1", "title": "France", "sentences": ["Paris is the capital city of France."]},
        ),
        Document(
            page_content="Berlin is the capital city of Germany.",
            metadata={"doc_id": "2", "title": "Germany", "sentences": ["Berlin is the capital city of Germany."]},
        ),
        Document(
            page_content="Madrid is the capital city of Spain.",
            metadata={"doc_id": "3", "title": "Spain", "sentences": ["Madrid is the capital city of Spain."]},
        ),
    ]


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        vectors = []
        for text in texts:
            text_lower = text.lower()
            vectors.append(
                [
                    float("france" in text_lower or "paris" in text_lower),
                    float("germany" in text_lower or "berlin" in text_lower),
                    float("spain" in text_lower or "madrid" in text_lower),
                ]
            )
        import numpy as np

        array = np.array(vectors, dtype="float32")
        if normalize_embeddings:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            array = array / norms
        return array


def test_bm25_returns_ranked_documents():
    retriever = build_bm25_retriever(_sample_documents(), top_k=2)
    results = retriever.retrieve("capital of France", top_k=2)
    assert len(results) == 2
    assert results[0].metadata["title"] == "France"


def test_faiss_retriever_builds_and_queries(monkeypatch):
    monkeypatch.setattr(retrieval_module, "SentenceTransformer", FakeSentenceTransformer)
    retriever = build_faiss_retriever(
        _sample_documents(),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=2,
        cache_dir=Path(".cache/test-faiss"),
    )
    results = retriever.retrieve("capital of Germany", top_k=2)
    assert results
    assert all("title" in document.metadata for document in results)


def test_hybrid_retriever_deduplicates_and_scores(monkeypatch):
    monkeypatch.setattr(retrieval_module, "SentenceTransformer", FakeSentenceTransformer)
    documents = _sample_documents()
    bm25 = build_bm25_retriever(documents, top_k=3)
    dense = build_faiss_retriever(
        documents,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
        cache_dir=Path(".cache/test-faiss"),
    )
    hybrid = build_hybrid_retriever(bm25, dense, alpha=0.5, top_k=2)
    results = hybrid.retrieve_with_scores("capital of Spain", top_k=2)
    assert len(results) == 2
    assert len({doc.metadata["doc_id"] for doc, _ in results}) == 2


def test_iterative_retriever_builds_from_config(monkeypatch):
    monkeypatch.setattr(retrieval_module, "SentenceTransformer", FakeSentenceTransformer)
    retriever = build_retriever(
        _sample_documents(),
        {
            "method": "iterative",
            "base_method": "hybrid",
            "top_k": 2,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_dir": Path(".cache/test-faiss"),
            "max_iter": 3,
            "sim_threshold": 0.5,
        },
    )
    results = retriever.retrieve_with_scores("capital of France", top_k=2)
    assert results
    assert len(results) <= 2

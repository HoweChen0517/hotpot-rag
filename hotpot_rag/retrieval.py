from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from pathlib import Path
import pickle
import re
from typing import Protocol

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if np.isclose(minimum, maximum):
        return {key: 1.0 for key in scores}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


class RetrieverLike(Protocol):
    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        ...

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        ...


@dataclass
class BM25Retriever:
    documents: list[Document]
    top_k: int = 10

    def __post_init__(self) -> None:
        self.corpus_tokens = [_tokenize(document.page_content) for document in self.documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        query_tokens = _tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked_indices = np.argsort(scores)[::-1][:k]
        return [(self.documents[index], float(scores[index])) for index in ranked_indices]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query=query, top_k=top_k)]


@dataclass
class DenseRetriever:
    documents: list[Document]
    embedding_model: str
    top_k: int = 10
    cache_dir: str | Path = ".cache/faiss"

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = SentenceTransformer(self.embedding_model)
        self.index, self.embedding_matrix = self._load_or_build_index()

    def _cache_key(self) -> str:
        joined_ids = "|".join(document.metadata.get("doc_id", str(idx)) for idx, document in enumerate(self.documents))
        digest = md5(f"{self.embedding_model}:{joined_ids}".encode("utf-8")).hexdigest()
        return digest

    def _cache_paths(self) -> tuple[Path, Path]:
        key = self._cache_key()
        return self.cache_dir / f"{key}.faiss", self.cache_dir / f"{key}.pkl"

    def _load_or_build_index(self) -> tuple[faiss.IndexFlatIP, np.ndarray]:
        index_path, meta_path = self._cache_paths()
        if index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            with meta_path.open("rb") as handle:
                payload = pickle.load(handle)
            return index, payload["embeddings"]

        texts = [document.page_content for document in self.documents]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        with meta_path.open("wb") as handle:
            pickle.dump({"embeddings": embeddings}, handle)
        return index, embeddings

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, k)
        results: list[tuple[Document, float]] = []
        for index, score in zip(indices[0], scores[0], strict=True):
            if index < 0:
                continue
            results.append((self.documents[index], float(score)))
        return results

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query=query, top_k=top_k)]


@dataclass
class HybridRetriever:
    bm25_retriever: BM25Retriever
    dense_retriever: DenseRetriever
    alpha: float = 0.5
    top_k: int = 10

    def retrieve_with_scores(
        self, query: str, top_k: int | None = None
    ) -> list[tuple[Document, float]]:
        k = top_k or self.top_k
        bm25_results = self.bm25_retriever.retrieve_with_scores(query=query, top_k=max(k, self.top_k))
        dense_results = self.dense_retriever.retrieve_with_scores(query=query, top_k=max(k, self.top_k))

        bm25_scores = {
            doc.metadata["doc_id"]: score for doc, score in bm25_results
        }
        dense_scores = {
            doc.metadata["doc_id"]: score for doc, score in dense_results
        }
        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_dense = _normalize_scores(dense_scores)

        doc_lookup = {
            doc.metadata["doc_id"]: doc
            for doc, _ in bm25_results + dense_results
        }
        combined_scores: dict[str, float] = {}
        for doc_id, document in doc_lookup.items():
            bm25_score = normalized_bm25.get(doc_id, 0.0)
            dense_score = normalized_dense.get(doc_id, 0.0)
            combined_scores[doc_id] = self.alpha * bm25_score + (1.0 - self.alpha) * dense_score

        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:k]
        return [(doc_lookup[doc_id], score) for doc_id, score in ranked]

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        return [doc for doc, _ in self.retrieve_with_scores(query=query, top_k=top_k)]


# TODO: More Retriever/Strategy to build



def build_bm25_retriever(documents: list[Document], top_k: int = 10) -> BM25Retriever:
    return BM25Retriever(documents=documents, top_k=top_k)


def build_faiss_retriever(
    documents: list[Document],
    embedding_model: str,
    top_k: int = 10,
    cache_dir: str | Path = ".cache/faiss",
) -> DenseRetriever:
    return DenseRetriever(
        documents=documents,
        embedding_model=embedding_model,
        top_k=top_k,
        cache_dir=cache_dir,
    )


def build_hybrid_retriever(
    bm25: BM25Retriever,
    dense: DenseRetriever,
    alpha: float = 0.5,
    top_k: int = 10,
) -> HybridRetriever:
    return HybridRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        alpha=alpha,
        top_k=top_k,
    )

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


@dataclass
class CrossEncoderReranker:
    model_name: str

    def __post_init__(self) -> None:
        self.model = CrossEncoder(self.model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:
        if not documents:
            return []
        pairs = [[query, document.page_content] for document in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
        return [(document, float(score)) for document, score in ranked[:top_k]]


def build_reranker(model_name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(model_name=model_name)

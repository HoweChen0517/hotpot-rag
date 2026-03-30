from langchain_core.documents import Document

import hotpot_rag.rerank as rerank_module
from hotpot_rag.rerank import build_reranker


class FakeCrossEncoder:
    def __init__(self, model_name):
        self.model_name = model_name

    def predict(self, pairs):
        scores = []
        for query, text in pairs:
            query_lower = query.lower()
            text_lower = text.lower()
            score = float(any(token in text_lower for token in query_lower.split()))
            scores.append(score)
        return scores


def test_reranker_reorders_documents(monkeypatch):
    monkeypatch.setattr(rerank_module, "CrossEncoder", FakeCrossEncoder)
    reranker = build_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    docs = [
        Document(page_content="Berlin is in Germany.", metadata={"title": "Germany"}),
        Document(page_content="Paris is in France.", metadata={"title": "France"}),
    ]
    ranked = reranker.rerank("Paris France", docs, top_k=1)
    assert ranked[0][0].metadata["title"] == "France"

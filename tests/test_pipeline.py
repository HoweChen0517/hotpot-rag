from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from hotpot_rag.data import HotpotSample
from hotpot_rag.pipeline import run_rag_case


class DummyRetriever:
    def retrieve_with_scores(self, query: str, top_k: int | None = None):
        return [
            (
                Document(
                    page_content="France\n\nParis is the capital city of France.",
                    metadata={"title": "France", "sentences": ["Paris is the capital city of France."]},
                ),
                1.0,
            )
        ]


class DummyReranker:
    def rerank(self, query, documents, top_k=5):
        return [(documents[0], 2.0)]


@dataclass
class DummyLLM:
    answer: str = "Paris"

    def invoke(self, messages, config=None, **kwargs):
        return AIMessage(content=self.answer)


def test_run_rag_case_returns_structured_output():
    sample = HotpotSample(
        sample_id="1",
        question="What is the capital of France?",
        answer="Paris",
        qtype="bridge",
        level="easy",
        supporting_facts=[{"title": "France", "sent_id": 0}],
        documents=[],
    )
    result = run_rag_case(
        sample,
        DummyRetriever(),
        DummyLLM(),
        reranker=DummyReranker(),
        retrieval_method="bm25",
        embedding_model="mini",
        used_model="dummy/model",
    )
    assert result["predicted_answer"] == "Paris"
    assert result["retrieved_titles"] == ["France"]
    assert result["rerank_enabled"] is True

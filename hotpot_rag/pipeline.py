from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from hotpot_rag.data import HotpotSample


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer user's questions using only the provided evidence. "
            "If the evidence is insufficient, respond with 'insufficient information'. "
            "Do not reveal chain-of-thought. Return only the final concise answer.",
        ),
        (
            "human",
            "Question:\n{question}\n\nEvidence:\n{context}\n\nFinal answer:",
        ),
    ]
)


def _format_context(documents: list[Document]) -> str:
    chunks = []
    for index, document in enumerate(documents, start=1):
        title = document.metadata.get("title", f"Document {index}")
        chunks.append(f"[{index}] {title}\n{document.page_content}")
    return "\n\n".join(chunks)


def run_rag_case(
    sample: HotpotSample,
    retriever: Any,
    llm: Any,
    reranker: Any | None = None,
    *,
    top_k_retrieve: int = 10,
    top_k_after_rerank: int = 5,
    retrieval_method: str = "unknown",
    embedding_model: str | None = None,
    used_model: str | None = None,
) -> dict:
    retrieved_with_scores = retriever.retrieve_with_scores(sample.question, top_k=top_k_retrieve)
    retrieved_docs = [document for document, _ in retrieved_with_scores]

    if reranker is not None:
        reranked = reranker.rerank(sample.question, retrieved_docs, top_k=top_k_after_rerank)
        final_docs = [document for document, _ in reranked]
    else:
        final_docs = retrieved_docs[:top_k_after_rerank]

    prompt_value = ANSWER_PROMPT.invoke(
        {"question": sample.question, "context": _format_context(final_docs)}
    )
    if isinstance(llm, Runnable):
        raw_response = llm.invoke(prompt_value)
    else:
        raw_response = llm.invoke(prompt_value.to_messages())

    # print(f"Question: {sample.question}")
    # print(f"Gold Answer: {sample.answer}")
    # print(f"Retrieved {len(retrieved_docs)} documents, reranked to {len(final_docs)} documents.")

    if isinstance(raw_response, BaseMessage):
        predicted_answer = str(raw_response.content).strip()
    else:
        predicted_answer = str(raw_response).strip()

    # print(f"Raw LLM Response: {raw_response}")
    # print(f"Predicted Answer: {predicted_answer}")

    return {
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold_answer": sample.answer,
        "predicted_answer": predicted_answer,
        "retrieved_titles": [document.metadata.get("title") for document in final_docs],
        "retrieved_contexts": [document.page_content for document in final_docs],
        "retrieved_sentences": [document.metadata.get("sentences", []) for document in final_docs],
        "supporting_facts": sample.supporting_facts,
        "used_model": used_model,
        "retrieval_method": retrieval_method,
        "embedding_model": embedding_model,
        "rerank_enabled": reranker is not None,
    }

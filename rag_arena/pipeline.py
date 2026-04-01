from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from rag_arena.data import ArenaSample


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


def _extract_response_text(raw_response: Any) -> str:
    if isinstance(raw_response, BaseMessage):
        content = raw_response.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
            return "\n".join(part for part in parts if part).strip()
        return str(content).strip()
    return str(raw_response).strip()


def run_rag_case(
    sample: ArenaSample,
    retriever: Any,
    llm: Any,
    reranker: Any | None = None,
    *,
    retriever_config: dict | None = None,
    rerank_config: dict | None = None,
    generation_config: dict | None = None,
) -> dict:
    retriever_config = retriever_config or {}
    rerank_config = rerank_config or {}
    generation_config = generation_config or {}

    top_k_retrieve = int(retriever_config.get("top_k", 10))
    top_k_after_rerank = int(rerank_config.get("top_k", 5))

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

    predicted_answer = _extract_response_text(raw_response)

    return {
        "dataset_name": sample.dataset_name,
        "sample_id": sample.sample_id,
        "question": sample.question,
        "gold_answer": sample.answer,
        "predicted_answer": predicted_answer,
        "retrieved_titles": [document.metadata.get("title") for document in final_docs],
        "retrieved_contexts": [document.page_content for document in final_docs],
        "retrieved_sentences": [document.metadata.get("sentences", []) for document in final_docs],
        "supporting_facts": sample.supporting_facts,
        "used_model": f"{generation_config.get('provider', 'unknown')}/{generation_config.get('model_name', 'unknown')}",
        "retrieval_method": retriever_config.get("method"),
        "embedding_model": retriever_config.get("embedding_model"),
        "rerank_enabled": reranker is not None,
    }

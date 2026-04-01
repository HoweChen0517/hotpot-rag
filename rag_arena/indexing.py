from __future__ import annotations

from langchain_core.documents import Document

from rag_arena.data import ArenaSample


def build_corpus(samples: list[ArenaSample]) -> list[Document]:
    documents: list[Document] = []
    for sample in samples:
        for document in sample.documents:
            metadata = dict(document.metadata)
            metadata["doc_id"] = document.doc_id
            metadata["sentences"] = document.sentences
            documents.append(Document(page_content=document.text, metadata=metadata))
    return documents

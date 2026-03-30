from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from datasets import Dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(slots=True)
class HotpotDocumentRecord:
    doc_id: str
    title: str
    text: str
    sentences: list[str]
    metadata: dict


@dataclass(slots=True)
class HotpotSample:
    sample_id: str
    question: str
    answer: str
    qtype: str
    level: str
    supporting_facts: list[dict]
    documents: list[HotpotDocumentRecord]


def _default_data_files(config: str, split: str) -> str:
    return str(PROJECT_ROOT / "hotpot_qa" / config / f"{split}-*.parquet")


def _to_sample(row: dict) -> HotpotSample:
    supporting_titles = set(row["supporting_facts"]["title"])
    supporting_facts = [
        {"title": title, "sent_id": sent_id}
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=True,
        )
    ]

    documents: list[HotpotDocumentRecord] = []
    for title, sentences in zip(
        row["context"]["title"],
        row["context"]["sentences"],
        strict=True,
    ):
        clean_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        if not title or not clean_sentences:
            continue
        doc_id = f"{row['id']}:{title}"
        documents.append(
            HotpotDocumentRecord(
                doc_id=doc_id,
                title=title,
                text=f"{title}\n\n{' '.join(clean_sentences)}",
                sentences=clean_sentences,
                metadata={
                    "sample_id": row["id"],
                    "title": title,
                    "is_supporting_doc": title in supporting_titles,
                },
            )
        )

    return HotpotSample(
        sample_id=row["id"],
        question=row["question"],
        answer=row["answer"],
        qtype=row["type"],
        level=row["level"],
        supporting_facts=supporting_facts,
        documents=documents,
    )


def load_hotpotqa_split(
    config: str,
    split: str,
    sample_size: int = 200,
    seed: int = 42,
    data_files: str | None = None,
) -> list[HotpotSample]:
    data_pattern = data_files or _default_data_files(config=config, split=split)
    if not Path(data_pattern).is_absolute():
        data_pattern = str((PROJECT_ROOT / data_pattern).resolve())
    dataset_dict = load_dataset("parquet", data_files={split: data_pattern})
    dataset: Dataset = dataset_dict[split]

    if sample_size and sample_size < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        dataset = dataset.select(indices[:sample_size])

    return [_to_sample(row) for row in dataset]

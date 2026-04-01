from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from datasets import Dataset, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ALIASES = {
    "hotpotqa": "hotpotqa",
    "hotpot_qa": "hotpotqa",
    "2wiki": "2wikimultihopqa",
    "2wikimultihopqa": "2wikimultihopqa",
    "framolfese/2wikimultihopqa": "2wikimultihopqa",
    "framolfese/2WikiMultihopQA": "2wikimultihopqa",
}

DATASET_DEFAULT_CONFIGS = {
    "hotpotqa": "distractor",
    "2wikimultihopqa": "default",
}


@dataclass(slots=True)
class ArenaDocumentRecord:
    doc_id: str
    title: str
    text: str
    sentences: list[str]
    metadata: dict


@dataclass(slots=True)
class ArenaSample:
    dataset_name: str
    sample_id: str
    question: str
    answer: str
    qtype: str
    level: str
    supporting_facts: list[dict]
    documents: list[ArenaDocumentRecord]


HotpotDocumentRecord = ArenaDocumentRecord
HotpotSample = ArenaSample


def canonicalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip()
    lowered = key.lower()
    if lowered in DATASET_ALIASES:
        return DATASET_ALIASES[lowered]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _default_data_files(dataset_name: str, dataset_config: str, split: str) -> str:
    canonical_name = canonicalize_dataset_name(dataset_name)
    if canonical_name == "hotpotqa":
        return str(PROJECT_ROOT / "hotpot_qa" / dataset_config / f"{split}-*.parquet")
    if canonical_name == "2wikimultihopqa":
        return str(PROJECT_ROOT / "2WikiMultihopQA" / "data" / f"{split}-*.parquet")
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _to_sample(row: dict, dataset_name: str) -> ArenaSample:
    supporting_titles = set(row["supporting_facts"]["title"])
    supporting_facts = [
        {"title": title, "sent_id": sent_id}
        for title, sent_id in zip(
            row["supporting_facts"]["title"],
            row["supporting_facts"]["sent_id"],
            strict=True,
        )
    ]

    documents: list[ArenaDocumentRecord] = []
    for title, sentences in zip(
        row["context"]["title"],
        row["context"]["sentences"],
        strict=True,
    ):
        clean_sentences = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]
        if not title or not clean_sentences:
            continue
        doc_id = f"{row['id']}:{title}"
        documents.append(
            ArenaDocumentRecord(
                doc_id=doc_id,
                title=title,
                text=f"{title}\n\n{' '.join(clean_sentences)}",
                sentences=clean_sentences,
                metadata={
                    "dataset_name": dataset_name,
                    "sample_id": row["id"],
                    "title": title,
                    "is_supporting_doc": title in supporting_titles,
                },
            )
        )

    return ArenaSample(
        dataset_name=dataset_name,
        sample_id=row["id"],
        question=row["question"],
        answer=row["answer"],
        qtype=row.get("type", "unknown"),
        level=row.get("level", "unknown"),
        supporting_facts=supporting_facts,
        documents=documents,
    )


def load_qa_split(
    dataset_name: str,
    split: str,
    sample_size: int = 200,
    seed: int = 42,
    dataset_config: str | None = None,
    data_files: str | None = None,
) -> list[ArenaSample]:
    canonical_name = canonicalize_dataset_name(dataset_name)
    resolved_config = dataset_config or DATASET_DEFAULT_CONFIGS[canonical_name]
    data_pattern = data_files or _default_data_files(
        dataset_name=canonical_name,
        dataset_config=resolved_config,
        split=split,
    )
    if not Path(data_pattern).is_absolute():
        data_pattern = str((PROJECT_ROOT / data_pattern).resolve())

    dataset_dict = load_dataset("parquet", data_files={split: data_pattern})
    dataset: Dataset = dataset_dict[split]

    if sample_size and sample_size < len(dataset):
        indices = list(range(len(dataset)))
        random.Random(seed).shuffle(indices)
        dataset = dataset.select(indices[:sample_size])

    return [_to_sample(row, canonical_name) for row in dataset]


def load_hotpotqa_split(
    config: str,
    split: str,
    sample_size: int = 200,
    seed: int = 42,
    data_files: str | None = None,
) -> list[ArenaSample]:
    return load_qa_split(
        dataset_name="hotpotqa",
        dataset_config=config,
        split=split,
        sample_size=sample_size,
        seed=seed,
        data_files=data_files,
    )


def load_2wikimultihopqa_split(
    split: str,
    sample_size: int = 200,
    seed: int = 42,
    data_files: str | None = None,
) -> list[ArenaSample]:
    return load_qa_split(
        dataset_name="2wikimultihopqa",
        split=split,
        sample_size=sample_size,
        seed=seed,
        data_files=data_files,
    )

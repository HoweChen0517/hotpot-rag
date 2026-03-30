from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from hotpot_rag.data import HotpotSample, load_hotpotqa_split
from hotpot_rag.evaluation import evaluate_predictions
from hotpot_rag.indexing import build_corpus
from hotpot_rag.llm import build_llm
from hotpot_rag.pipeline import run_rag_case
from hotpot_rag.rerank import build_reranker
from hotpot_rag.retrieval import (
    build_bm25_retriever,
    build_faiss_retriever,
    build_hybrid_retriever,
)


DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "google/embeddinggemma-300m",
    "intfloat/e5-small-v2",
]
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(slots=True)
class ExperimentConfig:
    dataset_config: str = "distractor"
    dataset_split: str = "validation"
    sample_size: int = 200
    seed: int = 42
    embedding_model: str = DEFAULT_EMBEDDING_MODELS[0]
    retrieval_method: str = "bm25"
    rerank_enabled: bool = False
    reranker_model: str = DEFAULT_RERANKER
    provider: str = "ollama"
    model_name: str = "qwen2.5:7b-instruct"
    temperature: float = 0.0
    max_tokens: int = 8192
    top_k_retrieve: int = 10
    top_k_after_rerank: int = 5
    alpha: float = 0.5
    cache_dir: str = ".cache/faiss"
    output_dir: str = "outputs/default"


@dataclass(slots=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics_path: str
    predictions_path: str
    summary: dict


def _build_retriever(samples: list[HotpotSample], config: ExperimentConfig):
    documents = build_corpus(samples)
    bm25 = build_bm25_retriever(documents, top_k=config.top_k_retrieve)

    if config.retrieval_method == "bm25":
        return bm25

    dense = build_faiss_retriever(
        documents,
        embedding_model=config.embedding_model,
        top_k=config.top_k_retrieve,
        cache_dir=config.cache_dir,
    )
    if config.retrieval_method == "dense":
        return dense
    if config.retrieval_method == "hybrid":
        return build_hybrid_retriever(
            bm25,
            dense,
            alpha=config.alpha,
            top_k=config.top_k_retrieve,
        )
    raise ValueError(f"Unsupported retrieval method: {config.retrieval_method}")


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_hotpotqa_split(
        config=config.dataset_config,
        split=config.dataset_split,
        sample_size=config.sample_size,
        seed=config.seed,
    )
    retriever = _build_retriever(samples, config)
    reranker = build_reranker(config.reranker_model) if config.rerank_enabled else None
    llm = build_llm(
        provider=config.provider,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    predictions = [
        run_rag_case(
            sample,
            retriever,
            llm,
            reranker=reranker,
            top_k_retrieve=config.top_k_retrieve,
            top_k_after_rerank=config.top_k_after_rerank,
            retrieval_method=config.retrieval_method,
            embedding_model=config.embedding_model,
            used_model=f"{config.provider}/{config.model_name}",
        )
        for sample in samples
    ]

    metrics_df = evaluate_predictions(predictions)
    summary = {
        "num_samples": len(metrics_df),
        "exact_match": float(metrics_df["exact_match"].mean()),
        "answer_f1": float(metrics_df["answer_f1"].mean()),
        "retrieval_recall_at_k": float(metrics_df["retrieval_recall_at_k"].mean()),
        "supporting_title_f1": float(metrics_df["supporting_title_f1"].mean()),
        "supporting_sentence_f1": float(metrics_df["supporting_sentence_f1"].mean()),
    }

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")

    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    config_path = output_dir / "experiment_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return ExperimentResult(
        config=config,
        metrics_path=str(metrics_path),
        predictions_path=str(predictions_path),
        summary=summary,
    )

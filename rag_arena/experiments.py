from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from rag_arena.data import ArenaSample, load_qa_split
from rag_arena.evaluation import evaluate_predictions, exact_match_score
from rag_arena.indexing import build_corpus
from rag_arena.llm import build_llm
from rag_arena.pipeline import run_rag_case
from rag_arena.rerank import build_reranker
from rag_arena.retrieval import build_retriever
from tqdm.auto import tqdm


DEFAULT_EMBEDDING_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "google/embeddinggemma-300m",
    "intfloat/e5-small-v2",
]
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def default_retriever_config() -> dict:
    return {
        "method": "bm25",
        "top_k": 10,
        "embedding_model": DEFAULT_EMBEDDING_MODELS[0],
        "alpha": 0.5,
        "base_method": "hybrid",
        "max_iter": 3,
        "sim_threshold": 0.85,
    }


def default_rerank_config() -> dict:
    return {
        "enabled": False,
        "model_name": DEFAULT_RERANKER,
        "top_k": 5,
    }


def default_generation_config() -> dict:
    return {
        "provider": "ollama",
        "model_name": "qwen2.5:7b-instruct",
        "temperature": 0.0,
        "max_tokens": 8192,
    }


def _merge_config(user_config: dict | None, default_factory) -> dict:
    merged = default_factory()
    if user_config:
        merged.update(user_config)
    return merged


@dataclass(slots=True)
class ExperimentConfig:
    dataset_name: str = "hotpotqa"
    dataset_config: str = "distractor"
    dataset_split: str = "validation"
    sample_size: int = 200
    seed: int = 42
    data_files: str | None = None
    cache_dir: str = ".cache/faiss"
    output_dir: str = "outputs/default"
    retriever_config: dict = field(default_factory=default_retriever_config)
    rerank_config: dict = field(default_factory=default_rerank_config)
    generation_config: dict = field(default_factory=default_generation_config)


@dataclass(slots=True)
class ExperimentResult:
    config: ExperimentConfig
    metrics_path: str
    predictions_path: str
    summary: dict


def _build_retriever(samples: list[ArenaSample], config: ExperimentConfig):
    documents = build_corpus(samples)
    retriever_config = _merge_config(config.retriever_config, default_retriever_config)
    retriever_config.setdefault("cache_dir", config.cache_dir)
    return build_retriever(documents, retriever_config)


def _is_retrieval_correct(prediction: dict) -> bool:
    gold_titles = {item["title"] for item in prediction["supporting_facts"]}
    predicted_titles = set(prediction.get("retrieved_titles", []))
    if not gold_titles:
        return True
    return gold_titles.issubset(predicted_titles)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    retriever_config = _merge_config(config.retriever_config, default_retriever_config)
    retriever_config.setdefault("cache_dir", config.cache_dir)
    rerank_config = _merge_config(config.rerank_config, default_rerank_config)
    generation_config = _merge_config(config.generation_config, default_generation_config)

    samples = load_qa_split(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        sample_size=config.sample_size,
        seed=config.seed,
        data_files=config.data_files,
    )
    retriever = build_retriever(build_corpus(samples), retriever_config)
    reranker = build_reranker(rerank_config)
    llm = build_llm(**generation_config)

    predictions = []
    retrieval_correct_count = 0
    exact_match_count = 0

    progress = tqdm(
        samples,
        desc=f"Running {config.dataset_name}",
        unit="sample",
        dynamic_ncols=True,
    )
    for sample in progress:
        prediction = run_rag_case(
            sample,
            retriever,
            llm,
            reranker=reranker,
            retriever_config=retriever_config,
            rerank_config=rerank_config,
            generation_config=generation_config,
        )
        predictions.append(prediction)

        if _is_retrieval_correct(prediction):
            retrieval_correct_count += 1
        if exact_match_score(prediction["predicted_answer"], prediction["gold_answer"]) == 1.0:
            exact_match_count += 1

        progress.set_postfix(
            retrieval_correct=f"{retrieval_correct_count}/{len(predictions)}",
            exact_match=f"{exact_match_count}/{len(predictions)}",
        )

    metrics_df = evaluate_predictions(predictions)
    summary = {
        "dataset_name": config.dataset_name,
        "num_samples": len(metrics_df),
        "retrieval_correct_count": retrieval_correct_count,
        "exact_match_count": exact_match_count,
        "exact_match": float(metrics_df["exact_match"].mean()),
        "answer_f1": float(metrics_df["answer_f1"].mean()),
        f"retrieval_recall@{rerank_config.get('top_k', 5) if rerank_config.get('enabled', False) else retriever_config.get('top_k', 5)}": float(metrics_df["supporting_title_recall@k"].mean()),
        f"retrieval_mrr@{rerank_config.get('top_k', 5) if rerank_config.get('enabled', False) else retriever_config.get('top_k', 5)}": float(metrics_df["supporting_title_mrr@k"].mean()),
        # "supporting_title_precision": float(metrics_df["supporting_title_precision@k"].mean()),
        # "supporting_title_recall": float(metrics_df["supporting_title_recall@k"].mean()),
        # "supporting_title_f1": float(metrics_df["supporting_title_f1@k"].mean()),
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

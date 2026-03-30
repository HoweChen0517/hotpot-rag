from __future__ import annotations

import argparse
import json

from hotpot_rag.experiments import DEFAULT_EMBEDDING_MODELS, ExperimentConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HotpotQA RAG experiments.")
    parser.add_argument("--dataset-config", default="distractor")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODELS[0])
    parser.add_argument("--retrieval-method", choices=["bm25", "dense", "hybrid"], default="bm25")
    parser.add_argument("--rerank-enabled", action="store_true")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama")
    parser.add_argument("--model-name", default="qwen2.5:7b-instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-k-retrieve", type=int, default=10)
    parser.add_argument("--top-k-after-rerank", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--cache-dir", default=".cache/faiss")
    parser.add_argument("--output-dir", default="outputs/default")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_experiment(ExperimentConfig(**vars(args)))
    print(json.dumps(result.summary, indent=2))

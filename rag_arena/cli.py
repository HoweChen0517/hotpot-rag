from __future__ import annotations

import argparse
import json

from rag_arena.experiments import ExperimentConfig, run_experiment


def _parse_json_argument(value: str) -> dict:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Rag Arena experiments.")
    parser.add_argument("--dataset-name", default="hotpotqa")
    parser.add_argument("--dataset-config", default="distractor")
    parser.add_argument("--dataset-split", default="validation")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-files", default=None)
    parser.add_argument("--cache-dir", default=".cache/faiss")
    parser.add_argument("--output-dir", default="outputs/default")
    parser.add_argument("--retriever-config", default="{}")
    parser.add_argument("--rerank-config", default="{}")
    parser.add_argument("--generation-config", default="{}")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        sample_size=args.sample_size,
        seed=args.seed,
        data_files=args.data_files,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        retriever_config=_parse_json_argument(args.retriever_config),
        rerank_config=_parse_json_argument(args.rerank_config),
        generation_config=_parse_json_argument(args.generation_config),
    )
    result = run_experiment(config)
    print(json.dumps(result.summary, indent=2))

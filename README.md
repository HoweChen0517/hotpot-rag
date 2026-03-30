# HotpotQA RAG Experiment Framework

This repository contains a LangChain-based RAG baseline for `HotpotQA`, focused on repeatable experiments over:

- embedding models
- retrieval methods (`BM25`, `Dense`, `Hybrid`) and more to-do...
- optional `cross-encoder` reranking
- multiple LLM backends (`Ollama`, `OpenRouter`)

## Install

```bash
uv sync
```

## Run a Smoke Experiment

```bash
uv run hotpot-rag \
  --sample-size 5 \
  --retrieval-method bm25 \
  --provider ollama \
  --model-name qwen2.5:7b-instruct \
  --output-dir outputs/smoke
```

## Environment Variables

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OPENROUTER_API_KEY=...
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

## Notebook

The main notebook is [notebooks/hotpotqa_rag_experiments.ipynb](/Volumes/HowesT7/NTU-Course-Materials/Assignments/AI6130%20-%20Large%20Language%20Models/AI6130-GroupProject/notebooks/hotpotqa_rag_experiments.ipynb).

It demonstrates:

- loading `HotpotQA distractor validation`
- building retrievers and rerankers
- running comparison experiments
- aggregating metrics and inspecting failure cases

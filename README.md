# Rag Arena

Rag Arena is a LangChain-based experimentation framework for multi-hop RAG evaluation.
It is designed to compare retrieval, reranking, and generation choices across multiple QA datasets with a consistent experiment runner.

## Core Modules

- `rag_arena.data`: dataset registry and loaders for `HotpotQA` and `2WikiMultihopQA`
- `rag_arena.indexing`: corpus construction from dataset contexts
- `rag_arena.retrieval`: sparse, dense, hybrid, and iterative retrievers
- `rag_arena.rerank`: reranker builders and cross-encoder reranking
- `rag_arena.llm`: LLM factories for `Ollama` and `OpenRouter`
- `rag_arena.pipeline`: end-to-end RAG case execution
- `rag_arena.experiments`: experiment config, runner, and result export
- `rag_arena.evaluation`: answer and supporting-fact metrics

## Install

```bash
uv sync
```

## Run

```bash
uv run rag-arena \
  --dataset-name hotpotqa \
  --retriever-config '{"method":"hybrid","top_k":10,"embedding_model":"sentence-transformers/all-MiniLM-L6-v2"}' \
  --rerank-config '{"enabled":true,"model_name":"cross-encoder/ms-marco-MiniLM-L-6-v2","top_k":5}' \
  --generation-config '{"provider":"ollama","model_name":"qwen2.5:7b-instruct","temperature":0.0,"max_tokens":8192}'
```

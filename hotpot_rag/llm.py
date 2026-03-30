from __future__ import annotations

import os
from typing import Any


def build_llm(provider: str, model_name: str, **kwargs: Any):
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", 8192)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            reasoning=False,    # do not reasoning
            base_url=base_url,
            **kwargs,
        )

    if provider == "openrouter":    # TODO - fixme: not the way to call openrouter
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter experiments.")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")

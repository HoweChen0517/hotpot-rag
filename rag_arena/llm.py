from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_api_key_from_json(key_name: str) -> str | None:
    api_keys_path = PROJECT_ROOT / "api_keys.json"
    if not api_keys_path.exists():
        return None
    with api_keys_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    value = payload.get(key_name)
    return str(value) if value else None


def _resolve_openrouter_api_key() -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    api_key = _load_api_key_from_json("OPENROUTER_API_KEY")
    if api_key:
        return api_key
    raise ValueError("OPENROUTER_API_KEY is required for OpenRouter experiments.")


def build_llm(provider: str, model_name: str, **kwargs: Any):
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", 8192)
    print(f"⏳Initializing {provider} model...")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        print(f"✅{provider} model loaded！")
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            num_predict=max_tokens,
            reasoning=False,    # do not reasoning
            base_url=base_url,
            **kwargs,
        )

    if provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        api_key = _resolve_openrouter_api_key()
        print(f"✅{provider} model loaded！")
        return ChatOpenRouter(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            reasoning={'enabled': False},    # do not reasoning
            **kwargs,
        )

    raise ValueError(f"Unsupported provider: {provider}")

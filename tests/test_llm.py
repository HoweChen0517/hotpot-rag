import json
import sys
import types

import rag_arena.llm as llm_module
from rag_arena.llm import build_llm


def test_build_ollama_llm(monkeypatch):
    module = types.ModuleType("langchain_ollama")

    class FakeChatOllama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatOllama = FakeChatOllama
    monkeypatch.setitem(sys.modules, "langchain_ollama", module)
    llm = build_llm("ollama", "qwen2.5:7b-instruct", temperature=0.1, max_tokens=128)
    assert llm.kwargs["model"] == "qwen2.5:7b-instruct"
    assert llm.kwargs["num_predict"] == 128


def test_build_openrouter_llm(monkeypatch):
    module = types.ModuleType("langchain_openrouter")

    class FakeChatOpenRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatOpenRouter = FakeChatOpenRouter
    monkeypatch.setitem(sys.modules, "langchain_openrouter", module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = build_llm("openrouter", "openai/gpt-4o-mini", temperature=0.1, max_tokens=128)
    assert llm.kwargs["model"] == "openai/gpt-4o-mini"
    assert llm.kwargs["api_key"] == "test-key"


def test_build_openrouter_llm_reads_api_key_from_json(monkeypatch, tmp_path):
    module = types.ModuleType("langchain_openrouter")

    class FakeChatOpenRouter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatOpenRouter = FakeChatOpenRouter
    monkeypatch.setitem(sys.modules, "langchain_openrouter", module)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(llm_module, "PROJECT_ROOT", tmp_path)
    (tmp_path / "api_keys.json").write_text(
        json.dumps({"OPENROUTER_API_KEY": "json-key"}),
        encoding="utf-8",
    )

    llm = build_llm("openrouter", "openai/gpt-4o-mini")
    assert llm.kwargs["api_key"] == "json-key"

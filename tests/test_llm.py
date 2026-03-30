import sys
import types

from hotpot_rag.llm import build_llm


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
    module = types.ModuleType("langchain_openai")

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    llm = build_llm("openrouter", "openai/gpt-4o-mini", temperature=0.1, max_tokens=128)
    assert llm.kwargs["model"] == "openai/gpt-4o-mini"
    assert llm.kwargs["api_key"] == "test-key"

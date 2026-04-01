from pathlib import Path

from rag_arena.data import load_2wikimultihopqa_split, load_hotpotqa_split, load_qa_split


def test_load_hotpotqa_split_parses_documents():
    samples = load_hotpotqa_split("distractor", "validation", sample_size=2, seed=0)
    assert len(samples) == 2
    first = samples[0]
    assert first.dataset_name == "hotpotqa"
    assert first.question
    assert first.documents
    assert all(document.title for document in first.documents)


def test_load_2wiki_split_parses_documents():
    samples = load_2wikimultihopqa_split("validation", sample_size=2, seed=0)
    assert len(samples) == 2
    first = samples[0]
    assert first.dataset_name == "2wikimultihopqa"
    assert first.level == "unknown"
    assert first.documents


def test_load_qa_split_is_independent_of_cwd(monkeypatch):
    monkeypatch.chdir(Path("notebooks"))
    samples = load_qa_split("hotpotqa", "validation", dataset_config="distractor", sample_size=1, seed=0)
    assert len(samples) == 1

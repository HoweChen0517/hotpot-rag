from pathlib import Path

from hotpot_rag.data import load_hotpotqa_split


def test_load_hotpotqa_split_parses_documents():
    samples = load_hotpotqa_split("distractor", "validation", sample_size=2, seed=0)
    assert len(samples) == 2
    first = samples[0]
    assert first.question
    assert first.answer is not None
    assert first.documents
    assert all(document.title for document in first.documents)
    assert all(document.text for document in first.documents)


def test_load_hotpotqa_split_is_independent_of_cwd(monkeypatch):
    monkeypatch.chdir(Path("notebooks"))
    samples = load_hotpotqa_split("distractor", "validation", sample_size=1, seed=0)
    assert len(samples) == 1

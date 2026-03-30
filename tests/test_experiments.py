from pathlib import Path

from hotpot_rag.experiments import ExperimentConfig, run_experiment


def test_run_experiment_writes_outputs(monkeypatch, tmp_path: Path):
    class DummyLLM:
        def invoke(self, messages, config=None, **kwargs):
            from langchain_core.messages import AIMessage

            return AIMessage(content="Paris")

    monkeypatch.setattr("hotpot_rag.experiments.build_llm", lambda **kwargs: DummyLLM())
    monkeypatch.setattr("hotpot_rag.experiments.load_hotpotqa_split", lambda **kwargs: [
        __import__("hotpot_rag.data", fromlist=["HotpotSample"]).HotpotSample(
            sample_id="1",
            question="What is the capital of France?",
            answer="Paris",
            qtype="bridge",
            level="easy",
            supporting_facts=[{"title": "France", "sent_id": 0}],
            documents=[
                __import__("hotpot_rag.data", fromlist=["HotpotDocumentRecord"]).HotpotDocumentRecord(
                    doc_id="1:France",
                    title="France",
                    text="France\n\nParis is the capital city of France.",
                    sentences=["Paris is the capital city of France."],
                    metadata={"sample_id": "1", "title": "France", "is_supporting_doc": True},
                )
            ],
        )
    ])

    result = run_experiment(
        ExperimentConfig(
            sample_size=1,
            retrieval_method="bm25",
            output_dir=str(tmp_path),
        )
    )
    assert result.summary["exact_match"] == 1.0
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "predictions.jsonl").exists()

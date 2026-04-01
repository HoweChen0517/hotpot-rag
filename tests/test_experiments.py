from pathlib import Path

from rag_arena.experiments import ExperimentConfig, run_experiment


def test_run_experiment_writes_outputs(monkeypatch, tmp_path: Path):
    class DummyLLM:
        def invoke(self, messages, config=None, **kwargs):
            from langchain_core.messages import AIMessage

            return AIMessage(content="Paris")

    monkeypatch.setattr("rag_arena.experiments.build_llm", lambda **kwargs: DummyLLM())
    monkeypatch.setattr("rag_arena.experiments.load_qa_split", lambda **kwargs: [
        __import__("rag_arena.data", fromlist=["ArenaSample"]).ArenaSample(
            dataset_name="hotpotqa",
            sample_id="1",
            question="What is the capital of France?",
            answer="Paris",
            qtype="bridge",
            level="easy",
            supporting_facts=[{"title": "France", "sent_id": 0}],
            documents=[
                __import__("rag_arena.data", fromlist=["ArenaDocumentRecord"]).ArenaDocumentRecord(
                    doc_id="1:France",
                    title="France",
                    text="France\n\nParis is the capital city of France.",
                    sentences=["Paris is the capital city of France."],
                    metadata={"dataset_name": "hotpotqa", "sample_id": "1", "title": "France", "is_supporting_doc": True},
                )
            ],
        )
    ])

    result = run_experiment(
        ExperimentConfig(
            sample_size=1,
            output_dir=str(tmp_path),
            retriever_config={"method": "bm25", "top_k": 5},
            generation_config={"provider": "ollama", "model_name": "dummy"},
        )
    )
    assert result.summary["exact_match"] == 1.0
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "predictions.jsonl").exists()

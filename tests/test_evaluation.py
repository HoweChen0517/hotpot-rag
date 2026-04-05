from rag_arena.evaluation import evaluate_predictions


def test_evaluate_predictions_outputs_expected_columns():
    df = evaluate_predictions(
        [
            {
                "dataset_name": "hotpotqa",
                "sample_id": "1",
                "gold_answer": "Paris",
                "predicted_answer": "Paris",
                "retrieved_titles": ["France"],
                "retrieved_sentences": [["Paris is the capital city of France."]],
                "supporting_facts": [{"title": "France", "sent_id": 0}],
                "retrieval_method": "bm25",
                "embedding_model": "mini",
                "used_model": "ollama/qwen2.5:7b-instruct",
                "rerank_enabled": False,
            }
        ]
    )
    assert df.loc[0, "exact_match"] == 1.0
    assert df.loc[0, "answer_f1"] == 1.0
    assert df.loc[0, "retrieval_mrr"] == 1.0
    assert df.loc[0, "supporting_title_recall"] == 1.0

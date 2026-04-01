from __future__ import annotations

from collections import Counter
import re

import pandas as pd


ARTICLES = {"a", "an", "the"}
PUNCT_RE = re.compile(r"[^a-z0-9 ]")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = PUNCT_RE.sub(" ", text)
    tokens = [token for token in text.split() if token not in ARTICLES]
    return " ".join(tokens)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(overlap.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _supporting_title_metrics(prediction: dict) -> tuple[float, float, float]:
    gold_titles = {item["title"] for item in prediction["supporting_facts"]}
    predicted_titles = set(prediction.get("retrieved_titles", []))
    if not gold_titles:
        return 1.0, 1.0, 1.0
    true_positive = len(gold_titles & predicted_titles)
    precision = true_positive / len(predicted_titles) if predicted_titles else 0.0
    recall = true_positive / len(gold_titles) if gold_titles else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _supporting_sentence_metrics(prediction: dict) -> tuple[float, float, float]:
    gold_pairs = {(item["title"], item["sent_id"]) for item in prediction["supporting_facts"]}
    predicted_pairs: set[tuple[str, int]] = set()
    for title, sentences in zip(
        prediction.get("retrieved_titles", []),
        prediction.get("retrieved_sentences", []),
        strict=False,
    ):
        for idx, _ in enumerate(sentences):
            predicted_pairs.add((title, idx))
    if not gold_pairs:
        return 1.0, 1.0, 1.0
    true_positive = len(gold_pairs & predicted_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(gold_pairs) if gold_pairs else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_predictions(predictions: list[dict]) -> pd.DataFrame:
    rows = []
    for prediction in predictions:
        title_precision, title_recall, title_f1 = _supporting_title_metrics(prediction)
        sentence_precision, sentence_recall, sentence_f1 = _supporting_sentence_metrics(prediction)
        rows.append(
            {
                "dataset_name": prediction.get("dataset_name"),
                "sample_id": prediction["sample_id"],
                "exact_match": exact_match_score(prediction["predicted_answer"], prediction["gold_answer"]),
                "answer_f1": f1_score(prediction["predicted_answer"], prediction["gold_answer"]),
                "retrieval_recall_at_k": title_recall,
                "supporting_title_precision": title_precision,
                "supporting_title_recall": title_recall,
                "supporting_title_f1": title_f1,
                "supporting_sentence_precision": sentence_precision,
                "supporting_sentence_recall": sentence_recall,
                "supporting_sentence_f1": sentence_f1,
                "retrieval_method": prediction.get("retrieval_method"),
                "embedding_model": prediction.get("embedding_model"),
                "used_model": prediction.get("used_model"),
                "rerank_enabled": prediction.get("rerank_enabled"),
            }
        )
    return pd.DataFrame(rows)

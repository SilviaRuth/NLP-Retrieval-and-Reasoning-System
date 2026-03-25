from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Iterable[int] | None = None,
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(int(gold == pred) for gold, pred in zip(y_true, y_pred))
    accuracy = correct / total if total else 0.0

    per_label = {}
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for label in labels:
        tp = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred == label)
        fp = sum(1 for gold, pred in zip(y_true, y_pred) if gold != label and pred == label)
        fn = sum(1 for gold, pred in zip(y_true, y_pred) if gold == label and pred != label)
        support = Counter(y_true)[label]

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1

        per_label[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    divisor = max(len(labels), 1)
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision / divisor,
        "macro_recall": macro_recall / divisor,
        "macro_f1": macro_f1 / divisor,
        "per_label": per_label,
    }


def recall_at_k(ranked_ids: Sequence[Sequence[str]], relevant_ids: Sequence[set[str]], k: int) -> float:
    if len(ranked_ids) != len(relevant_ids):
        raise ValueError("ranked_ids and relevant_ids must have the same length")

    hits = 0
    for ranking, relevant in zip(ranked_ids, relevant_ids):
        if set(ranking[:k]) & relevant:
            hits += 1
    return hits / len(ranked_ids) if ranked_ids else 0.0


def mean_reciprocal_rank(ranked_ids: Sequence[Sequence[str]], relevant_ids: Sequence[set[str]]) -> float:
    if len(ranked_ids) != len(relevant_ids):
        raise ValueError("ranked_ids and relevant_ids must have the same length")

    reciprocal_ranks = []
    for ranking, relevant in zip(ranked_ids, relevant_ids):
        rr = 0.0
        for index, doc_id in enumerate(ranking, start=1):
            if doc_id in relevant:
                rr = 1.0 / index
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

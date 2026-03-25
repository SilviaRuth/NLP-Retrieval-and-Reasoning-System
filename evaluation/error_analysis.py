from __future__ import annotations

from collections import Counter

from data.datasets import NLIExample
from utils.text import contains_negation, lexical_overlap_ratio, simple_tokenize


def categorize_error(example: NLIExample) -> str:
    premise_len = len(simple_tokenize(example.premise))
    hypothesis_len = len(simple_tokenize(example.hypothesis))

    if contains_negation(example.premise) != contains_negation(example.hypothesis):
        return "negation"
    if lexical_overlap_ratio(example.premise, example.hypothesis) > 0.6:
        return "lexical_overlap"
    if max(premise_len, hypothesis_len) > 30:
        return "long_sequence"
    return "other"


def analyze_errors(
    examples: list[NLIExample],
    y_true: list[int],
    y_pred: list[int],
    id_to_label: dict[int, str],
) -> dict:
    records = []
    categories = Counter()
    for example, gold, pred in zip(examples, y_true, y_pred):
        if gold == pred:
            continue
        category = categorize_error(example)
        categories[category] += 1
        records.append(
            {
                "premise": example.premise,
                "hypothesis": example.hypothesis,
                "gold": id_to_label[gold],
                "predicted": id_to_label[pred],
                "category": category,
            }
        )

    return {
        "total_errors": len(records),
        "category_counts": dict(categories),
        "records": records,
    }

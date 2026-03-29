from __future__ import annotations

from collections import Counter

from data.datasets import NLIExample
from evaluation.hard_set import extract_error_tags
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
    source_split: str = "validation",
) -> dict:
    records = []
    categories = Counter()
    tag_counts = Counter()
    for index, (example, gold, pred) in enumerate(zip(examples, y_true, y_pred)):
        if gold == pred:
            continue
        category = categorize_error(example)
        tags = extract_error_tags(example.premise, example.hypothesis, category)
        categories[category] += 1
        tag_counts.update(tags)
        records.append(
            {
                "example_id": index,
                "premise": example.premise,
                "hypothesis": example.hypothesis,
                "gold": id_to_label[gold],
                "gold_label": id_to_label[gold],
                "predicted": id_to_label[pred],
                "predicted_label": id_to_label[pred],
                "category": category,
                "error_tags": tags,
                "source_split": source_split,
            }
        )

    return {
        "total_errors": len(records),
        "category_counts": dict(categories),
        "tag_counts": dict(tag_counts),
        "records": records,
    }

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable

from data import NLIExample, load_nli_dataset
from evaluation.metrics import classification_metrics
from utils import contains_negation, read_json, simple_tokenize, write_json


MONTH_NAMES = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
TEMPORAL_TERMS = {
    "before",
    "after",
    "during",
    "since",
    "until",
    "later",
    "earlier",
    "century",
    "year",
    "month",
    "week",
    "day",
    "born",
    "died",
    "released",
    "premiered",
    "founded",
    "first",
    "last",
    "previous",
    "next",
}
ORDINAL_PATTERN = re.compile(r"\b\d+(st|nd|rd|th)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"\b\d+[\d,./-]*\b")


@dataclass
class HardSetExample:
    premise: str
    hypothesis: str
    gold_label: str
    predicted_label: str
    error_tags: list[str]
    source_split: str
    example_id: str


PRIORITY_BUCKETS = ("negation", "long_sequence", "numeric_date")


def _has_numeric_or_temporal_reasoning(text: str) -> bool:
    lowered = text.lower()
    tokens = set(simple_tokenize(lowered))
    return bool(NUMBER_PATTERN.search(lowered) or ORDINAL_PATTERN.search(lowered) or tokens & MONTH_NAMES or tokens & TEMPORAL_TERMS)


def extract_error_tags(premise: str, hypothesis: str, primary_category: str | None = None) -> list[str]:
    tags: list[str] = []
    if primary_category and primary_category not in {"", "other"}:
        tags.append(primary_category)

    premise_tokens = simple_tokenize(premise)
    hypothesis_tokens = simple_tokenize(hypothesis)
    if contains_negation(premise) or contains_negation(hypothesis):
        tags.append("negation")
    if max(len(premise_tokens), len(hypothesis_tokens)) > 30:
        tags.append("long_sequence")
    if _has_numeric_or_temporal_reasoning(premise) or _has_numeric_or_temporal_reasoning(hypothesis):
        tags.append("numeric_date")

    deduped: list[str] = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped or ["other"]


def find_latest_error_analysis(root: str | Path = "outputs") -> Path:
    candidates = sorted(Path(root).glob("**/error_analysis.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No error_analysis.json files were found under outputs/")
    return candidates[0]


def _index_examples(examples: list[NLIExample]) -> dict[tuple[str, str, str], list[int]]:
    index: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for position, example in enumerate(examples):
        index[(example.premise, example.hypothesis, example.label)].append(position)
    return index


def _resolve_example_id(record: dict, source_index: dict[tuple[str, str, str], list[int]]) -> str:
    if record.get("example_id") is not None:
        return str(record["example_id"])

    key = (record["premise"], record["hypothesis"], record.get("gold_label", record.get("gold", "")))
    positions = source_index.get(key, [])
    if positions:
        return str(positions.pop(0))
    return "unmatched"


def build_hard_set(
    error_analysis_path: str | Path,
    source_path: str | Path,
    output_path: str | Path,
    source_split: str = "validation",
) -> dict:
    payload = read_json(error_analysis_path)
    source_examples = load_nli_dataset(source_path)
    source_index = _index_examples(source_examples)

    hard_examples: list[HardSetExample] = []
    for record in payload.get("records", []):
        tags = record.get("error_tags") or extract_error_tags(
            record["premise"],
            record["hypothesis"],
            record.get("category"),
        )
        if not any(tag in PRIORITY_BUCKETS for tag in tags):
            continue
        hard_examples.append(
            HardSetExample(
                premise=record["premise"],
                hypothesis=record["hypothesis"],
                gold_label=record.get("gold_label", record.get("gold", "")),
                predicted_label=record.get("predicted_label", record.get("predicted", "")),
                error_tags=tags,
                source_split=record.get("source_split", source_split),
                example_id=_resolve_example_id(record, source_index),
            )
        )

    bucket_counts = {bucket: sum(1 for example in hard_examples if bucket in example.error_tags) for bucket in PRIORITY_BUCKETS}
    output_payload = {
        "source_error_analysis": str(error_analysis_path),
        "source_split": source_split,
        "num_examples": len(hard_examples),
        "bucket_counts": bucket_counts,
        "examples": [asdict(example) for example in hard_examples],
    }
    write_json(output_path, output_payload)
    return output_payload


def load_hard_set(path: str | Path) -> dict:
    return read_json(path)


def compute_bucket_metrics(
    examples: Iterable[dict],
    y_true: list[int],
    y_pred: list[int],
    label_to_id: dict[str, int],
) -> dict:
    examples = list(examples)
    summary: dict[str, dict] = {}
    for bucket in PRIORITY_BUCKETS:
        indices = [index for index, example in enumerate(examples) if bucket in example.get("error_tags", [])]
        if not indices:
            summary[bucket] = {"count": 0, "accuracy": 0.0, "macro_f1": 0.0}
            continue
        gold_subset = [y_true[index] for index in indices]
        pred_subset = [y_pred[index] for index in indices]
        metrics = classification_metrics(gold_subset, pred_subset, labels=label_to_id.values())
        summary[bucket] = {
            "count": len(indices),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
        }
    return summary

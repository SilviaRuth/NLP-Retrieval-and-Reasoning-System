from __future__ import annotations

from collections import Counter
import re
from typing import Iterable

from .text import contains_negation, lexical_overlap_ratio, simple_tokenize


ALLOWED_LABELS = {"entailment", "contradiction", "neutral"}
NEGATION_CATEGORIES = {"negation"}
NUMERIC_CATEGORIES = {"numeric", "numeric contradiction", "numeric_contradiction"}
TEMPORAL_CATEGORIES = {"temporal", "temporal/date reasoning", "temporal_date_reasoning"}
LONG_REASONING_CATEGORIES = {
    "long_reasoning",
    "long reasoning",
    "long-premise short-hypothesis reasoning",
}
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
TIME_PATTERN = re.compile(r"\b\d{1,2}:\d{2}\b")
DIGIT_PATTERN = re.compile(r"\d")
PLACEHOLDER_PATTERNS = [
    re.compile(r"<[^>]+>"),
    re.compile(r"\{[^}]+\}"),
    re.compile(r"\b(?:TBD|TODO|PLACEHOLDER|FILL_ME|FIXME|XXX)\b", re.IGNORECASE),
    re.compile(r"\b(?:premise_here|hypothesis_here|template_text)\b", re.IGNORECASE),
]
NUMERIC_QUANTIFIER_PHRASES = {
    "all",
    "any",
    "at least",
    "at most",
    "each",
    "every",
    "exactly",
    "fewer than",
    "less than",
    "more than",
    "no",
    "none",
    "not all",
    "some",
}
TEMPORAL_CUES = {
    "after",
    "before",
    "during",
    "earlier",
    "ended",
    "ending",
    "ends",
    "finished",
    "later",
    "same day",
    "started",
    "starting",
    "through",
    "until",
    "yesterday",
}


def normalize_category_name(category: str) -> str:
    lowered = str(category).strip().lower()
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def normalize_text(text: str) -> str:
    return " ".join(simple_tokenize(text))


def has_placeholder_artifact(text: str) -> bool:
    return any(pattern.search(text) for pattern in PLACEHOLDER_PATTERNS)


def has_numeric_signal(text: str) -> bool:
    lowered = text.lower()
    if DIGIT_PATTERN.search(lowered):
        return True
    return any(phrase in lowered for phrase in NUMERIC_QUANTIFIER_PHRASES)


def has_temporal_signal(text: str) -> bool:
    lowered = text.lower()
    if DATE_PATTERN.search(lowered) or TIME_PATTERN.search(lowered):
        return True
    return any(cue in lowered for cue in TEMPORAL_CUES)


def is_too_short(text: str, min_tokens: int) -> bool:
    return len(simple_tokenize(text)) < min_tokens


def is_repetitive(text: str) -> bool:
    tokens = [token for token in simple_tokenize(text) if any(char.isalnum() for char in token)]
    if len(tokens) < 6:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    most_common_count = Counter(tokens).most_common(1)[0][1]
    if unique_ratio < 0.45:
        return True
    if most_common_count / len(tokens) > 0.35:
        return True
    for index in range(len(tokens) - 2):
        if tokens[index] == tokens[index + 1] == tokens[index + 2]:
            return True
    return False


def is_semantically_broken(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped[-1] in {":", ";", "-"}:
        return True
    alnum_tokens = [token for token in simple_tokenize(stripped) if any(char.isalpha() for char in token)]
    return not alnum_tokens


def detect_duplicate_reason(example: dict, accepted_examples: list[dict]) -> str | None:
    premise = str(example["premise"])
    hypothesis = str(example["hypothesis"])
    label = str(example["label"])
    category = normalize_category_name(str(example["category"]))
    normalized_pair = (normalize_text(premise), normalize_text(hypothesis), label, category)

    for accepted in accepted_examples:
        accepted_pair = (
            normalize_text(str(accepted["premise"])),
            normalize_text(str(accepted["hypothesis"])),
            str(accepted["label"]),
            normalize_category_name(str(accepted["category"])),
        )
        if normalized_pair == accepted_pair:
            return "duplicate_exact"

        if label != accepted_pair[2] or category != accepted_pair[3]:
            continue

        premise_overlap = lexical_overlap_ratio(premise, str(accepted["premise"]))
        hypothesis_overlap = lexical_overlap_ratio(hypothesis, str(accepted["hypothesis"]))
        pair_overlap = lexical_overlap_ratio(
            f"{premise} {hypothesis}",
            f"{accepted['premise']} {accepted['hypothesis']}",
        )
        if premise_overlap >= 0.96 and hypothesis_overlap >= 0.96:
            return "duplicate_near"
        if pair_overlap >= 0.98:
            return "duplicate_near"

    return None


def validate_synthetic_nli_example(example: dict, accepted_examples: list[dict]) -> list[str]:
    reasons: list[str] = []
    required_fields = {"premise", "hypothesis", "label", "category"}
    missing_fields = sorted(required_fields.difference(example))
    if missing_fields:
        reasons.append(f"missing_required_fields:{','.join(missing_fields)}")
        return reasons

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = str(example["label"]).strip().lower()
    category = normalize_category_name(str(example["category"]))

    if not isinstance(premise, str) or not premise.strip():
        reasons.append("invalid_premise")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        reasons.append("invalid_hypothesis")
    if reasons:
        return reasons

    premise = premise.strip()
    hypothesis = hypothesis.strip()
    combined = f"{premise} {hypothesis}"

    if label not in ALLOWED_LABELS:
        reasons.append("invalid_label")
    if premise == hypothesis and label != "entailment":
        reasons.append("identical_text_invalid_label")
    if has_placeholder_artifact(combined):
        reasons.append("template_artifact")
    if is_too_short(premise, min_tokens=4):
        reasons.append("premise_too_short")
    if is_too_short(hypothesis, min_tokens=3):
        reasons.append("hypothesis_too_short")
    if is_repetitive(premise) or is_repetitive(hypothesis):
        reasons.append("repetitive_text")
    if is_semantically_broken(premise) or is_semantically_broken(hypothesis):
        reasons.append("semantically_broken")

    if category in NEGATION_CATEGORIES and not (contains_negation(premise) or contains_negation(hypothesis)):
        reasons.append("negation_missing_negation_logic")
    if category in NUMERIC_CATEGORIES and not (has_numeric_signal(premise) or has_numeric_signal(hypothesis)):
        reasons.append("numeric_missing_quantity_signal")
    if category in TEMPORAL_CATEGORIES and not (has_temporal_signal(premise) or has_temporal_signal(hypothesis)):
        reasons.append("temporal_missing_temporal_cue")
    if category in LONG_REASONING_CATEGORIES:
        if premise.count(".") < 2 and len(simple_tokenize(premise)) < 28:
            reasons.append("long_reasoning_premise_too_short")

    duplicate_reason = detect_duplicate_reason(example, accepted_examples)
    if duplicate_reason is not None:
        reasons.append(duplicate_reason)

    return reasons


def validate_synthetic_nli_examples(examples: Iterable[dict]) -> tuple[list[dict], list[dict], dict]:
    example_list = list(examples)
    accepted_examples: list[dict] = []
    rejected_examples: list[dict] = []
    rejection_counts: Counter[str] = Counter()

    for index, example in enumerate(example_list):
        copied_example = dict(example)
        reasons = validate_synthetic_nli_example(copied_example, accepted_examples)
        if reasons:
            copied_example["validation_status"] = "rejected"
            copied_example["rejection_reasons"] = reasons
            copied_example["rejection_index"] = index
            rejected_examples.append(copied_example)
            rejection_counts.update(reasons)
            continue

        copied_example["validation_status"] = "passed"
        accepted_examples.append(copied_example)

    report = {
        "total_generated": len(example_list),
        "total_accepted": len(accepted_examples),
        "total_rejected": len(rejected_examples),
        "rejection_reasons_by_count": dict(sorted(rejection_counts.items())),
    }
    return accepted_examples, rejected_examples, report

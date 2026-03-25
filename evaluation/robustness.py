from __future__ import annotations

from typing import Callable

from data.datasets import NLIExample
from evaluation.metrics import classification_metrics
from utils.text import paraphrase_text, shuffle_words, typo_noise


def _perturb(example: NLIExample, strategy: str, seed: int) -> NLIExample:
    if strategy == "typo":
        return NLIExample(
            premise=typo_noise(example.premise, seed=seed),
            hypothesis=typo_noise(example.hypothesis, seed=seed + 1),
            label=example.label,
        )
    if strategy == "paraphrase":
        return NLIExample(
            premise=paraphrase_text(example.premise),
            hypothesis=paraphrase_text(example.hypothesis),
            label=example.label,
        )
    if strategy == "shuffle":
        return NLIExample(
            premise=shuffle_words(example.premise, seed=seed),
            hypothesis=shuffle_words(example.hypothesis, seed=seed + 1),
            label=example.label,
        )
    raise ValueError(f"Unsupported robustness strategy: {strategy}")


def evaluate_robustness(
    examples: list[NLIExample],
    predict_fn: Callable[[list[NLIExample]], list[int]],
    label_to_id: dict[str, int],
    strategies: tuple[str, ...] = ("typo", "paraphrase", "shuffle"),
    seed: int = 42,
) -> dict:
    gold = [label_to_id[example.label] for example in examples]
    results = {}
    for offset, strategy in enumerate(strategies):
        transformed = [_perturb(example, strategy, seed + offset) for example in examples]
        predicted = predict_fn(transformed)
        results[strategy] = classification_metrics(gold, predicted, labels=label_to_id.values())
    return results

from __future__ import annotations

import argparse
from collections import defaultdict
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import read_json


LABEL_ORDER = ("entailment", "contradiction", "neutral")
DEFAULT_PER_CATEGORY = 12


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def derive_output_path(input_path: Path, explicit_output_path: str | None) -> Path:
    if explicit_output_path:
        return resolve_path(explicit_output_path)
    return input_path.with_name(f"review_sample_{input_path.stem}.md")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample generated NLI data for human review.")
    parser.add_argument("--input", required=True, help="Path to a generated dataset JSON file.")
    parser.add_argument("--output", help="Path to the review markdown file.")
    parser.add_argument(
        "--per-category",
        type=int,
        default=DEFAULT_PER_CATEGORY,
        help="Target number of sampled examples per category when available.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducible sampling.")
    return parser.parse_args(argv)


def load_examples(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return payload["examples"]
    raise ValueError(f"Unsupported synthetic dataset format in {path}")


def validate_example(example: dict) -> None:
    required_fields = {"premise", "hypothesis", "label", "category"}
    missing = sorted(required_fields - set(example))
    if missing:
        raise ValueError(f"Example missing required fields: {', '.join(missing)}")
    if example["label"] not in LABEL_ORDER:
        raise ValueError(f"Unsupported NLI label for review sampling: {example['label']}")


def group_examples_by_category_and_label(examples: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for example in examples:
        validate_example(example)
        grouped[example["category"]][example["label"]].append(example)
    return grouped


def sample_category_examples(
    grouped_examples: dict[str, list[dict]],
    per_category: int,
    rng: random.Random,
) -> list[dict]:
    label_pools: dict[str, list[dict]] = {}
    for label, examples in grouped_examples.items():
        shuffled_examples = list(examples)
        rng.shuffle(shuffled_examples)
        label_pools[label] = shuffled_examples

    available_labels = [label for label in LABEL_ORDER if label_pools.get(label)]
    if not available_labels:
        return []

    target = min(per_category, sum(len(examples) for examples in label_pools.values()))
    selected: list[dict] = []

    while len(selected) < target:
        made_progress = False
        for label in available_labels:
            pool = label_pools[label]
            if not pool:
                continue
            selected.append(pool.pop())
            made_progress = True
            if len(selected) >= target:
                break
        if not made_progress:
            break

    return selected


def sample_examples(examples: list[dict], per_category: int, seed: int) -> dict[str, list[dict]]:
    if per_category <= 0:
        raise ValueError("--per-category must be a positive integer")

    grouped = group_examples_by_category_and_label(examples)
    rng = random.Random(seed)
    sampled_by_category: dict[str, list[dict]] = {}
    for category in sorted(grouped):
        sampled_by_category[category] = sample_category_examples(grouped[category], per_category, rng)
    return sampled_by_category


def format_example_block(index: int, example: dict) -> str:
    generation_method = example.get("generation_method", "unknown")
    return "\n".join(
        [
            f"### Example {index}",
            f"- Category: {example['category']}",
            f"- Label: {example['label']}",
            f"- Generation method: {generation_method}",
            f"- Premise: {example['premise']}",
            f"- Hypothesis: {example['hypothesis']}",
            "",
        ]
    )


def render_review_markdown(
    sampled_by_category: dict[str, list[dict]],
    input_path: Path,
    seed: int,
    per_category: int,
) -> str:
    total_sampled = sum(len(examples) for examples in sampled_by_category.values())
    lines = [
        "# Synthetic NLI Review Sample",
        "",
        f"- Input file: `{input_path}`",
        f"- Sampling seed: `{seed}`",
        f"- Target examples per category: `{per_category}`",
        f"- Total sampled examples: `{total_sampled}`",
        "",
    ]

    for category, examples in sampled_by_category.items():
        label_counts = defaultdict(int)
        for example in examples:
            label_counts[example["label"]] += 1

        label_summary = ", ".join(
            f"{label}: {label_counts[label]}" for label in LABEL_ORDER if label_counts.get(label)
        )
        lines.extend(
            [
                f"## {category}",
                "",
                f"- Sampled examples: `{len(examples)}`",
                f"- Label mix: {label_summary or 'none'}",
                "",
            ]
        )

        for index, example in enumerate(examples, start=1):
            lines.append(format_example_block(index, example))

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = resolve_path(args.input)
    output_path = derive_output_path(input_path, args.output)

    examples = load_examples(input_path)
    sampled_by_category = sample_examples(examples, args.per_category, args.seed)
    review_markdown = render_review_markdown(sampled_by_category, input_path, args.seed, args.per_category)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review_markdown, encoding="utf-8")

    print(f"Sampled {sum(len(category_examples) for category_examples in sampled_by_category.values())} examples")
    print(f"Review file: {output_path}")


if __name__ == "__main__":
    main()

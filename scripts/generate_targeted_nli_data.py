from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import random
from datetime import date, timedelta

from utils import write_json


ENTITIES = ["device", "sensor", "archive", "projector", "tablet", "router"]
PLACES = ["lab", "museum", "warehouse", "station", "studio", "library"]
ANIMALS = ["fox", "otter", "falcon", "lynx", "whale", "badger"]
ADJECTIVES = ["connected", "active", "visible", "charged", "available", "open"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate targeted synthetic NLI data for reasoning-heavy failure buckets.")
    parser.add_argument("--negation", type=int, default=0)
    parser.add_argument("--numeric", type=int, default=0)
    parser.add_argument("--temporal", type=int, default=0)
    parser.add_argument("--long_reasoning", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-path", default="data/generated/targeted_nli_run3.json")
    return parser.parse_args()


def build_negation_example(rng: random.Random, index: int) -> dict:
    entity = rng.choice(ENTITIES)
    adjective = rng.choice(ADJECTIVES)
    premise = f"The {entity} is {adjective}."
    label_cycle = index % 3
    if label_cycle == 0:
        hypothesis = f"The {entity} is {adjective}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {entity} is not {adjective}."
        label = "contradiction"
    else:
        place = rng.choice(PLACES)
        hypothesis = f"The {entity} is stored in the {place}."
        label = "neutral"
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "category": "negation",
    }


def build_numeric_example(rng: random.Random, index: int) -> dict:
    animal = rng.choice(ANIMALS)
    quantity = rng.randint(2, 12)
    premise = f"There are {quantity} {animal}s near the river." if quantity != 1 else f"There is {quantity} {animal} near the river."
    label_cycle = index % 3
    if label_cycle == 0:
        hypothesis = f"There are at least {max(quantity - 1, 1)} {animal}s near the river."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"There are {quantity + 2} {animal}s near the river."
        label = "contradiction"
    else:
        hypothesis = f"The {animal}s were seen before sunrise."
        label = "neutral"
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "category": "numeric_date",
    }


def build_temporal_example(rng: random.Random, index: int) -> dict:
    start_date = date(2010, 1, 1) + timedelta(days=rng.randint(0, 3650))
    finish_date = start_date + timedelta(days=rng.randint(5, 40))
    premise = f"The festival started on {start_date.isoformat()} and ended on {finish_date.isoformat()}."
    label_cycle = index % 3
    if label_cycle == 0:
        hypothesis = f"The festival ended after {start_date.isoformat()}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The festival ended before {start_date.isoformat()}."
        label = "contradiction"
    else:
        hypothesis = f"The festival returned to the same city in 2030."
        label = "neutral"
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "category": "numeric_date",
    }


def build_long_reasoning_example(rng: random.Random, index: int) -> dict:
    animal = rng.choice(ANIMALS)
    place = rng.choice(PLACES)
    adjective = rng.choice(ADJECTIVES)
    quantity = rng.randint(3, 9)
    premise = (
        f"The report says the {animal} habitat survey was organized in the {place}. "
        f"Field teams counted {quantity} animals near the northern trail, noted that the tracking device remained {adjective}, "
        "and confirmed that no relocation took place during the final inspection window. "
        "The lead biologist signed the report after comparing the new notes with the earlier season archive."
    )
    label_cycle = index % 3
    if label_cycle == 0:
        hypothesis = f"The survey found {quantity} animals near the northern trail."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = "The animals were relocated before the final inspection window."
        label = "contradiction"
    else:
        hypothesis = f"The survey was funded by the {rng.choice(PLACES)} council."
        label = "neutral"
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "category": "long_reasoning",
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    examples: list[dict] = []
    category_counts: dict[str, int] = {}

    generators = [
        ("negation", args.negation, build_negation_example),
        ("numeric", args.numeric, build_numeric_example),
        ("temporal", args.temporal, build_temporal_example),
        ("long_reasoning", args.long_reasoning, build_long_reasoning_example),
    ]
    for category_name, count, generator in generators:
        for index in range(count):
            examples.append(generator(rng, index))
        category_counts[category_name] = count

    payload = {
        "summary": {
            "seed": args.seed,
            "total_examples": len(examples),
            "category_counts": category_counts,
        },
        "examples": examples,
    }
    write_json(args.output_path, payload)
    print(f"Saved {len(examples)} synthetic examples to {args.output_path}")
    for category_name, count in category_counts.items():
        print(f"- {category_name}: {count}")


if __name__ == "__main__":
    main()



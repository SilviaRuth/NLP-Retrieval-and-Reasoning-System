from __future__ import annotations

import argparse
from collections import Counter
from datetime import date, datetime, time, timedelta
import re
import sys
from pathlib import Path
import random
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import read_json, write_json
from utils.synthetic_nli_validation import validate_synthetic_nli_examples


LABELS = ("entailment", "contradiction", "neutral")
DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
GENERATION_METHOD = "diverse_targeted_nli_v2"
DEFAULT_OUTPUT = "data/generated/targeted_nli_v1.json"
PROTECTED_SOURCE_FILES = {
    (ROOT / "data" / "train.json").resolve(),
    (ROOT / "data" / "validation.json").resolve(),
    (ROOT / "data" / "test.json").resolve(),
}
CATEGORY_LABELS = {
    "negation": "negation",
    "numeric": "numeric contradiction",
    "temporal": "temporal/date reasoning",
    "long_reasoning": "long-premise short-hypothesis reasoning",
}
CATEGORY_SEED_OFFSETS = {
    "negation": 100_000,
    "numeric": 200_000,
    "temporal": 300_000,
    "long_reasoning": 400_000,
}
AMBIGUOUS_TERMS = {
    "about",
    "approximately",
    "around",
    "few",
    "likely",
    "many",
    "maybe",
    "might",
    "often",
    "perhaps",
    "possibly",
    "probably",
    "roughly",
    "several",
    "sometimes",
    "usually",
}
DOUBLE_NEGATION_PATTERNS = [
    re.compile(r"\bnot\s+\w+\s+not\b"),
    re.compile(r"\bnot impossible\b"),
    re.compile(r"\bnot uncommon\b"),
    re.compile(r"\bcannot not\b"),
    re.compile(r"\bcan't not\b"),
    re.compile(r"\bnever not\b"),
]

FIRST_NAMES = [
    "Ava",
    "Ben",
    "Clara",
    "Daniel",
    "Elena",
    "Felix",
    "Grace",
    "Hugo",
    "Iris",
    "Jonah",
    "Lena",
    "Mina",
    "Nora",
    "Omar",
    "Priya",
    "Rosa",
    "Theo",
    "Victor",
    "Yara",
    "Zane",
]
ROLES = [
    "archivist",
    "assistant",
    "coordinator",
    "curator",
    "engineer",
    "inspector",
    "librarian",
    "manager",
    "operator",
    "supervisor",
    "technician",
]
ENTITIES = [
    "alarm panel",
    "backup server",
    "camera",
    "control tablet",
    "delivery gate",
    "display screen",
    "east cabinet",
    "heater",
    "label printer",
    "loading dock door",
    "router",
    "sensor",
    "side window",
    "storage cabinet",
    "ticket scanner",
]
STATES = [
    "active",
    "connected",
    "locked",
    "open",
    "ready",
    "sealed",
    "visible",
    "working",
]
BROKEN_STATES = [
    "broken",
    "delayed",
    "empty",
    "jammed",
    "missing",
    "offline",
    "unlocked",
    "wet",
]
PLACES = [
    "archive room",
    "control booth",
    "east lab",
    "field office",
    "front desk",
    "library annex",
    "north station",
    "records office",
    "repair bay",
    "south warehouse",
    "training hall",
    "visitor lobby",
]
ITEM_TYPES = [
    "badge",
    "box",
    "crate",
    "folder",
    "form",
    "manual",
    "package",
    "sample",
    "ticket",
]
DOCUMENT_TYPES = [
    "claim form",
    "consent form",
    "invoice",
    "order sheet",
    "registration card",
    "survey form",
]
LOCKER_TYPES = [
    "cabinet",
    "drawer",
    "gate",
    "locker",
]
ANIMALS = [
    "badger",
    "crane",
    "falcon",
    "fox",
    "heron",
    "lynx",
    "otter",
    "seal",
    "whale",
    "wolf",
]
EVENTS = [
    "audit",
    "briefing",
    "inspection",
    "orientation",
    "review",
    "training session",
    "trial run",
    "workshop",
]
CONTAINERS = [
    "carton",
    "crate",
    "folder",
    "sample box",
    "tray",
]
COLORS = ["blue", "green", "red", "white", "yellow"]
WORK_AREAS = [
    "cold room",
    "east table",
    "front cart",
    "inspection bench",
    "packing desk",
    "side shelf",
    "west pallet",
]


def pluralize(noun: str) -> str:
    if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
        return f"{noun[:-1]}ies"
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return f"{noun}es"
    return f"{noun}s"


def load_generation_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    payload = read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Generation config at {config_path} must be a JSON object")
    if "output_path" in payload and "output" not in payload:
        payload["output"] = payload.pop("output_path")
    if "summary_path" in payload and "summary_output" not in payload:
        payload["summary_output"] = payload.pop("summary_path")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap_args, _ = bootstrap.parse_known_args(argv)
    config_defaults = load_generation_config(bootstrap_args.config)

    parser = argparse.ArgumentParser(description="Generate targeted synthetic NLI data without modifying the gold dataset files.")
    parser.add_argument("--config")
    parser.add_argument("--negation", type=int, default=0)
    parser.add_argument("--numeric", type=int, default=0)
    parser.add_argument("--temporal", type=int, default=0)
    parser.add_argument("--long_reasoning", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "--output-path", dest="output", default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-output")
    parser.add_argument("--rejected-output")
    parser.add_argument("--validation-report-output")
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)

    for field_name in ("negation", "numeric", "temporal", "long_reasoning"):
        if getattr(args, field_name) < 0:
            raise ValueError(f"{field_name} must be non-negative")

    return args

def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def format_output_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def derive_summary_path(output_path: Path, explicit_summary_path: str | None) -> Path:
    if explicit_summary_path:
        return resolve_path(explicit_summary_path)
    return output_path.with_name(f"{output_path.stem}_summary.json")


def ensure_safe_output_paths(output_path: Path, summary_path: Path) -> None:
    if output_path == summary_path:
        raise ValueError("Output path and summary path must be different files")
    for candidate in (output_path, summary_path):
        if candidate in PROTECTED_SOURCE_FILES:
            raise ValueError(f"Refusing to write to protected source file: {format_output_path(candidate)}")


def choose_distinct(rng: random.Random, pool: list[str], count: int) -> list[str]:
    return rng.sample(pool, count)


def random_date(rng: random.Random) -> date:
    return date(2021, 1, 1) + timedelta(days=rng.randint(0, 1400))


def random_time_pair(rng: random.Random) -> tuple[time, time]:
    hour_one = rng.randint(8, 14)
    minute_one = rng.choice((0, 10, 15, 20, 30, 40, 45, 50))
    delta_hours = rng.randint(1, 4)
    delta_minutes = rng.choice((0, 10, 15, 20, 30))
    first = time(hour_one, minute_one)
    second_dt = datetime.combine(date(2000, 1, 1), first) + timedelta(hours=delta_hours, minutes=delta_minutes)
    second = second_dt.time().replace(second=0, microsecond=0)
    return first, second


def time_str(value: time) -> str:
    return value.strftime("%H:%M")


def contains_ambiguous_language(text: str) -> bool:
    tokens = {token.strip(".,;:!?\"'()[]").lower() for token in text.split()}
    if tokens.intersection(AMBIGUOUS_TERMS):
        return True
    lowered = text.lower()
    return any(pattern.search(lowered) for pattern in DOUBLE_NEGATION_PATTERNS)


def build_example(
    *,
    premise: str,
    hypothesis: str,
    label: str,
    category_key: str,
    template_id: str,
    seed: int,
) -> dict:
    return {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": label,
        "source": "synthetic",
        "generation_method": GENERATION_METHOD,
        "category": CATEGORY_LABELS[category_key],
        "template_id": template_id,
        "seed": seed,
        "validation_status": "pending",
    }


def make_candidate(example: dict, validation: dict) -> dict:
    return {"example": example, "validation": validation}


def build_negation_direct(label_cycle: int, rng: random.Random, seed: int) -> dict:
    role = rng.choice(ROLES)
    entity = rng.choice(ENTITIES)
    state = rng.choice(STATES)
    place = rng.choice(PLACES)
    premise = rng.choice(
        [
            f"During the closing check, the {role} noted that the {entity} was not {state} in the {place}.",
            f"The maintenance log for the {place} says the {entity} was not {state}.",
            f"According to the inspection note, the {entity} was not {state} when the {role} reviewed the {place}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {entity} was not {state}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {entity} was {state}."
        label = "contradiction"
    else:
        hypothesis = f"The {entity} was replaced after the inspection."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="negation",
        template_id="negation_direct",
        seed=seed,
    )
    return make_candidate(example, {"category_key": "negation", "subtype": "direct", "entity": entity, "state": state})


def build_negation_scope(label_cycle: int, rng: random.Random, seed: int) -> dict:
    speaker = rng.choice(FIRST_NAMES)
    denied_entity, asserted_entity = choose_distinct(rng, ENTITIES, 2)
    denied_state = rng.choice(BROKEN_STATES)
    asserted_state = rng.choice(BROKEN_STATES)
    while asserted_state == denied_state:
        asserted_state = rng.choice(BROKEN_STATES)
    premise = rng.choice(
        [
            f"{speaker} did not say that the {denied_entity} was {denied_state}; {speaker} said that the {asserted_entity} was {asserted_state}.",
            f"In the briefing, {speaker} did not claim the {denied_entity} was {denied_state}. Instead, {speaker} said the {asserted_entity} was {asserted_state}.",
            f"The transcript shows that {speaker} did not say the {denied_entity} was {denied_state}; the recorded claim was that the {asserted_entity} was {asserted_state}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"{speaker} said the {asserted_entity} was {asserted_state}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"{speaker} said the {denied_entity} was {denied_state}."
        label = "contradiction"
    else:
        hypothesis = f"{speaker} repaired the {denied_entity}."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="negation",
        template_id="negation_scope",
        seed=seed,
    )
    return make_candidate(
        example,
        {
            "category_key": "negation",
            "subtype": "scope",
            "speaker": speaker,
            "denied_entity": denied_entity,
            "denied_state": denied_state,
            "asserted_entity": asserted_entity,
            "asserted_state": asserted_state,
        },
    )


def build_negation_not_all(label_cycle: int, rng: random.Random, seed: int) -> dict:
    total = rng.randint(4, 12)
    incomplete = rng.randint(1, total - 1)
    document = rng.choice(DOCUMENT_TYPES)
    documents = pluralize(document)
    premise = rng.choice(
        [
            f"Not all {total} {documents} were complete; {incomplete} still needed signatures.",
            f"Of the {total} {documents}, not all were complete because {incomplete} still needed signatures.",
            f"The review found that not all {total} {documents} were complete: {incomplete} still needed signatures.",
        ]
    )
    if label_cycle == 0:
        hypothesis = rng.choice(
            [
                f"Some {documents} were not complete.",
                f"Not every {document} was complete.",
            ]
        )
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"All {total} {documents} were complete."
        label = "contradiction"
    else:
        hypothesis = f"The incomplete {documents} were resubmitted the next morning."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="negation",
        template_id="negation_not_all",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "negation", "subtype": "not_all", "total": total, "incomplete": incomplete, "document": document},
    )


def build_negation_none(label_cycle: int, rng: random.Random, seed: int) -> dict:
    total = rng.randint(3, 10)
    fixture = rng.choice(LOCKER_TYPES)
    place = rng.choice(["west wall", "north corridor", "storage row", "service hall"])
    premise = rng.choice(
        [
            f"None of the {total} {pluralize(fixture)} on the {place} were open.",
            f"The audit found that none of the {total} {pluralize(fixture)} on the {place} were open.",
            f"According to the checklist, none of the {total} {pluralize(fixture)} on the {place} were open.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"No {fixture} on the {place} was open."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"At least one {fixture} on the {place} was open."
        label = "contradiction"
    else:
        hypothesis = f"The {pluralize(fixture)} on the {place} were repainted last month."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="negation",
        template_id="negation_none",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "negation", "subtype": "none", "total": total, "fixture": fixture, "place": place},
    )

def build_numeric_exact_count(label_cycle: int, rng: random.Random, seed: int) -> dict:
    quantity = rng.randint(4, 28)
    mismatch = quantity + rng.choice([2, 3, 4, 5])
    item = rng.choice(CONTAINERS)
    descriptor = rng.choice(COLORS)
    place = rng.choice(WORK_AREAS)
    items = f"{descriptor} {pluralize(item)}"
    premise = rng.choice(
        [
            f"The inventory sheet lists exactly {quantity} {items} on the {place}.",
            f"During the count, the team found exactly {quantity} {items} on the {place}.",
            f"The morning audit recorded exactly {quantity} {items} on the {place}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {place} held {quantity} {items}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {place} held {mismatch} {items}."
        label = "contradiction"
    else:
        hypothesis = f"Two of the {items} were opened that evening."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="numeric",
        template_id="numeric_exact_count",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "numeric", "subtype": "exact_count", "quantity": quantity, "mismatch": mismatch},
    )


def build_numeric_comparison(label_cycle: int, rng: random.Random, seed: int) -> dict:
    item = rng.choice(ITEM_TYPES)
    plural_item = pluralize(item)
    place_a, place_b = choose_distinct(rng, ["east room", "west room", "front shelf", "back shelf", "north rack", "south rack"], 2)
    count_a = rng.randint(5, 20)
    count_b = rng.randint(2, 18)
    while count_b == count_a:
        count_b = rng.randint(2, 18)
    premise = rng.choice(
        [
            f"The {place_a} stored {count_a} {plural_item}, while the {place_b} stored {count_b}.",
            f"In the final count, the {place_a} had {count_a} {plural_item} and the {place_b} had {count_b}.",
            f"The stock note says the {place_a} held {count_a} {plural_item}; the {place_b} held {count_b}.",
        ]
    )
    more_place = place_a if count_a > count_b else place_b
    less_place = place_b if count_a > count_b else place_a
    if label_cycle == 0:
        hypothesis = f"The {more_place} stored more {plural_item} than the {less_place}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {less_place} stored more {plural_item} than the {more_place}."
        label = "contradiction"
    else:
        hypothesis = f"Both areas were cleaned after the count."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="numeric",
        template_id="numeric_comparison",
        seed=seed,
    )
    return make_candidate(
        example,
        {
            "category_key": "numeric",
            "subtype": "comparison",
            "count_a": count_a,
            "count_b": count_b,
            "more_place": more_place,
            "less_place": less_place,
        },
    )


def build_numeric_bounds(label_cycle: int, rng: random.Random, seed: int) -> dict:
    quantity = rng.randint(6, 24)
    item = rng.choice(ANIMALS)
    plural_item = pluralize(item)
    place = rng.choice(["east marsh", "north trail", "riverbank", "south field"])
    premise = rng.choice(
        [
            f"The field report recorded exactly {quantity} {plural_item} near the {place} checkpoint.",
            f"During the survey, the team counted exactly {quantity} {plural_item} near the {place} checkpoint.",
            f"The observer's note says there were exactly {quantity} {plural_item} near the {place} checkpoint.",
        ]
    )
    if label_cycle == 0:
        if rng.choice([True, False]):
            lower_bound = max(1, quantity - rng.randint(1, 3))
            hypothesis = f"At least {lower_bound} {plural_item} were near the {place} checkpoint."
        else:
            upper_bound = quantity + rng.randint(1, 3)
            hypothesis = f"At most {upper_bound} {plural_item} were near the {place} checkpoint."
        label = "entailment"
    elif label_cycle == 1:
        if rng.choice([True, False]):
            too_high = quantity + rng.randint(2, 5)
            hypothesis = f"At least {too_high} {plural_item} were near the {place} checkpoint."
        else:
            too_low = max(0, quantity - rng.randint(2, 5))
            hypothesis = f"At most {too_low} {plural_item} were near the {place} checkpoint."
        label = "contradiction"
    else:
        hypothesis = f"The {plural_item} crossed the checkpoint before sunrise."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="numeric",
        template_id="numeric_bounds",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "numeric", "subtype": "bounds", "quantity": quantity},
    )


def build_numeric_parts_total(label_cycle: int, rng: random.Random, seed: int) -> dict:
    item = rng.choice(["folder", "manual", "sample box", "ticket"])
    plural_item = pluralize(item)
    place_a, place_b = choose_distinct(rng, ["front table", "side table", "north shelf", "south shelf"], 2)
    count_a = rng.randint(2, 11)
    count_b = rng.randint(2, 11)
    total = count_a + count_b
    mismatch = total + rng.choice([1, 2, 3])
    premise = rng.choice(
        [
            f"The {place_a} held {count_a} {plural_item}, and the {place_b} held {count_b} {plural_item}.",
            f"On the count sheet, the {place_a} had {count_a} {plural_item} while the {place_b} had {count_b}.",
            f"The stock note lists {count_a} {plural_item} on the {place_a} and {count_b} on the {place_b}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"There were {total} {plural_item} altogether."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"There were {mismatch} {plural_item} altogether."
        label = "contradiction"
    else:
        hypothesis = f"The {plural_item} were sorted by size after the count."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="numeric",
        template_id="numeric_parts_total",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "numeric", "subtype": "parts_total", "count_a": count_a, "count_b": count_b, "total": total, "mismatch": mismatch},
    )


def build_temporal_before_after(label_cycle: int, rng: random.Random, seed: int) -> dict:
    event_one, event_two = choose_distinct(rng, EVENTS, 2)
    date_one = random_date(rng)
    date_two = date_one + timedelta(days=rng.randint(1, 5))
    premise = rng.choice(
        [
            f"The {event_one} happened on {date_one.isoformat()}. The {event_two} happened on {date_two.isoformat()}.",
            f"Records show the {event_one} took place on {date_one.isoformat()}, and the {event_two} took place on {date_two.isoformat()}.",
            f"The schedule lists the {event_one} on {date_one.isoformat()} and the {event_two} on {date_two.isoformat()}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {event_one} happened before the {event_two}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {event_one} happened after the {event_two}."
        label = "contradiction"
    else:
        hypothesis = f"The {event_two} moved to another building."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="temporal",
        template_id="temporal_before_after",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "temporal", "subtype": "before_after", "date_one": date_one, "date_two": date_two},
    )


def build_temporal_same_day(label_cycle: int, rng: random.Random, seed: int) -> dict:
    event_one, event_two = choose_distinct(rng, ["debrief", "intake interview", "inspection", "rehearsal", "review call", "safety drill"], 2)
    event_date = random_date(rng)
    start_one, start_two = random_time_pair(rng)
    premise = rng.choice(
        [
            f"The {event_one} started at {time_str(start_one)} on {event_date.isoformat()}. The {event_two} started at {time_str(start_two)} on the same day.",
            f"On {event_date.isoformat()}, the {event_one} began at {time_str(start_one)} and the {event_two} began at {time_str(start_two)}.",
            f"The schedule for {event_date.isoformat()} shows the {event_one} at {time_str(start_one)} and the {event_two} at {time_str(start_two)}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {event_one} and the {event_two} happened on the same day."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {event_one} and the {event_two} happened on different days."
        label = "contradiction"
    else:
        hypothesis = f"The {event_two} lasted thirty minutes."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="temporal",
        template_id="temporal_same_day",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "temporal", "subtype": "same_day", "date": event_date},
    )


def build_temporal_duration(label_cycle: int, rng: random.Random, seed: int) -> dict:
    event = rng.choice(EVENTS)
    duration_days = rng.randint(2, 5)
    start_date = random_date(rng)
    end_date = start_date + timedelta(days=duration_days - 1)
    premise = rng.choice(
        [
            f"The {event} ran for {duration_days} days, from {start_date.isoformat()} through {end_date.isoformat()}.",
            f"According to the calendar, the {event} lasted {duration_days} days and ran from {start_date.isoformat()} through {end_date.isoformat()}.",
            f"The plan says the {event} lasted {duration_days} days, beginning on {start_date.isoformat()} and ending on {end_date.isoformat()}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {event} lasted more than one day."
        label = "entailment"
    elif label_cycle == 1:
        too_short = max(1, duration_days - 1)
        hypothesis = f"The {event} lasted {too_short} day."
        if too_short != 1:
            hypothesis = f"The {event} lasted {too_short} days."
        label = "contradiction"
    else:
        hypothesis = f"The {event} switched to an online format on the final day."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="temporal",
        template_id="temporal_duration",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "temporal", "subtype": "duration", "duration_days": duration_days},
    )


def build_temporal_sequence(label_cycle: int, rng: random.Random, seed: int) -> dict:
    person = rng.choice(FIRST_NAMES)
    event_date = random_date(rng)
    submitted_time, approved_time = random_time_pair(rng)
    approved_dt = datetime.combine(event_date, approved_time)
    printed_dt = approved_dt + timedelta(hours=rng.randint(1, 3), minutes=rng.choice((0, 15, 30)))
    premise = rng.choice(
        [
            f"{person} submitted the draft at {time_str(submitted_time)} on {event_date.isoformat()}. The editor approved it at {time_str(approved_time)}. The print team started at {time_str(printed_dt.time())}.",
            f"On {event_date.isoformat()}, {person} submitted the draft at {time_str(submitted_time)}. Approval came at {time_str(approved_time)}, and printing started at {time_str(printed_dt.time())}.",
            f"The timeline for {event_date.isoformat()} lists the draft submission at {time_str(submitted_time)}, approval at {time_str(approved_time)}, and printing at {time_str(printed_dt.time())}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = "The draft was approved after it was submitted."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = "The print team started before the draft was approved."
        label = "contradiction"
    else:
        hypothesis = "The draft was translated before printing."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="temporal",
        template_id="temporal_sequence",
        seed=seed,
    )
    return make_candidate(
        example,
        {
            "category_key": "temporal",
            "subtype": "sequence",
            "submitted_dt": datetime.combine(event_date, submitted_time),
            "approved_dt": approved_dt,
            "printed_dt": printed_dt,
        },
    )


def build_temporal_day_relation(label_cycle: int, rng: random.Random, seed: int) -> dict:
    event_one, event_two = choose_distinct(rng, ["equipment pickup", "final review", "follow-up call", "site visit", "training test"], 2)
    first_date = random_date(rng)
    later_date = first_date + timedelta(days=rng.randint(2, 6))
    premise = rng.choice(
        [
            f"The {event_one} happened on {first_date.isoformat()}. The {event_two} happened later, on {later_date.isoformat()}.",
            f"The log shows the {event_one} on {first_date.isoformat()} and the {event_two} on {later_date.isoformat()}.",
            f"Records list the {event_one} for {first_date.isoformat()} and the {event_two} for {later_date.isoformat()}.",
        ]
    )
    if label_cycle == 0:
        hypothesis = f"The {event_two} happened later than the {event_one}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The {event_one} and the {event_two} happened on the same day."
        label = "contradiction"
    else:
        hypothesis = f"The {event_two} lasted two hours."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="temporal",
        template_id="temporal_day_relation",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "temporal", "subtype": "day_relation", "first_date": first_date, "later_date": later_date},
    )

def build_long_reasoning_transfer(label_cycle: int, rng: random.Random, seed: int) -> dict:
    sealed_item = rng.choice(["sample box", "crate", "tray"])
    moved_place = rng.choice(["cold room", "inspection bench", "packing desk"])
    stayed_place = rng.choice(["loading bay", "side shelf", "west pallet"])
    count_good = rng.randint(8, 18)
    count_bad = rng.randint(2, 6)
    plural_item = pluralize(sealed_item)
    premise = (
        f"During the morning audit, the team counted {count_good} sealed {plural_item} on the front pallet and {count_bad} damaged ones near the door. "
        f"Only the sealed {plural_item} were moved to the {moved_place}; the damaged ones stayed by the {stayed_place} for inspection. "
        f"Before lunch, the supervisor signed the transfer sheet for the {plural_item} that went to the {moved_place}."
    )
    if label_cycle == 0:
        hypothesis = f"The sealed {plural_item} were moved to the {moved_place}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"The damaged {plural_item} were moved to the {moved_place}."
        label = "contradiction"
    else:
        hypothesis = f"The sealed {plural_item} were shipped to another building."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="long_reasoning",
        template_id="long_reasoning_transfer",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "long_reasoning", "subtype": "transfer", "plural_item": plural_item, "moved_place": moved_place},
    )


def build_long_reasoning_access(label_cycle: int, rng: random.Random, seed: int) -> dict:
    person_a, person_b, person_c = choose_distinct(rng, FIRST_NAMES, 3)
    place = rng.choice(["archive vault", "east lab", "records room", "testing room"])
    pass_one = rng.choice(["badge", "code card", "entry key"])
    pass_two = rng.choice(["visitor pass", "clearance note", "safety form"])
    premise = (
        f"Three visitors checked in for the {place}: {person_a}, {person_b}, and {person_c}. "
        f"{person_a} had the {pass_one}, {person_b} had the {pass_two}, and {person_c} had both. "
        f"The coordinator explained that entry required both documents. In the end, only {person_c} entered the {place}."
    )
    if label_cycle == 0:
        hypothesis = f"{person_c} entered the {place}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"{person_a} entered the {place}."
        label = "contradiction"
    else:
        hypothesis = f"{person_b} requested a temporary pass later."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="long_reasoning",
        template_id="long_reasoning_access",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "long_reasoning", "subtype": "access", "person_a": person_a, "person_c": person_c, "place": place},
    )


def build_long_reasoning_assignment(label_cycle: int, rng: random.Random, seed: int) -> dict:
    drafter, reviewer = choose_distinct(rng, FIRST_NAMES, 2)
    document = rng.choice(["incident summary", "inventory report", "site memo", "visit log"])
    premise = (
        f"{drafter} drafted the {document} before noon. "
        f"{reviewer} checked the figures but did not edit the {document}. "
        f"After the corrections were approved, {drafter} emailed the final {document} to the director."
    )
    if label_cycle == 0:
        hypothesis = f"{drafter} emailed the final {document}."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = f"{reviewer} emailed the final {document}."
        label = "contradiction"
    else:
        hypothesis = f"The director printed the {document} immediately."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="long_reasoning",
        template_id="long_reasoning_assignment",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "long_reasoning", "subtype": "assignment", "drafter": drafter, "reviewer": reviewer, "document": document},
    )


def build_long_reasoning_incident(label_cycle: int, rng: random.Random, seed: int) -> dict:
    person_a, person_b = choose_distinct(rng, FIRST_NAMES, 2)
    machine = rng.choice(["backup pump", "sorting belt", "ticket printer", "water filter"])
    failure_time = time(rng.randint(6, 9), rng.choice((0, 10, 15, 30, 45)))
    restart_time = (datetime.combine(date(2000, 1, 1), failure_time) + timedelta(hours=2, minutes=15)).time()
    premise = (
        f"The {machine} failed at {time_str(failure_time)}. "
        f"{person_a} shut off the line, and {person_b} replaced the clogged part. "
        f"Once the pressure returned to normal, {person_a} restarted the {machine} at {time_str(restart_time)}. "
        f"The supervisor wrote the incident report after the restart."
    )
    if label_cycle == 0:
        hypothesis = "The report was written after the restart."
        label = "entailment"
    elif label_cycle == 1:
        hypothesis = "The report was written before the restart."
        label = "contradiction"
    else:
        hypothesis = "A replacement machine arrived that afternoon."
        label = "neutral"
    example = build_example(
        premise=premise,
        hypothesis=hypothesis,
        label=label,
        category_key="long_reasoning",
        template_id="long_reasoning_incident",
        seed=seed,
    )
    return make_candidate(
        example,
        {"category_key": "long_reasoning", "subtype": "incident", "failure_time": failure_time, "restart_time": restart_time},
    )


def validate_generic_example(example: dict) -> str:
    required_fields = {
        "premise",
        "hypothesis",
        "label",
        "source",
        "generation_method",
        "category",
        "template_id",
        "seed",
        "validation_status",
    }
    missing_fields = required_fields.difference(example)
    if missing_fields:
        return f"failed:missing_{','.join(sorted(missing_fields))}"
    if example["label"] not in LABELS:
        return "failed:invalid_label"
    if example["source"] != "synthetic":
        return "failed:invalid_source"
    if example["generation_method"] != GENERATION_METHOD:
        return "failed:invalid_generation_method"
    premise = str(example["premise"]).strip()
    hypothesis = str(example["hypothesis"]).strip()
    if not premise or not hypothesis:
        return "failed:empty_text"
    combined = f"{premise} {hypothesis}"
    if contains_ambiguous_language(combined):
        return "failed:ambiguous_language"
    if premise == hypothesis:
        return "failed:premise_hypothesis_identical"
    return "passed"


def validate_negation(candidate: dict) -> str:
    context = candidate["validation"]
    example = candidate["example"]
    label = example["label"]
    hypothesis = example["hypothesis"].lower()
    premise = example["premise"].lower()
    subtype = context["subtype"]

    if subtype == "direct":
        if " not " not in premise:
            return "failed:negation_direct_missing_negation"
        if label == "entailment" and " not " not in hypothesis:
            return "failed:negation_entailment_should_be_negative"
        if label == "contradiction" and " not " in hypothesis:
            return "failed:negation_contradiction_should_flip_polarity"
        return "passed"

    if subtype == "scope":
        denied_phrase = f"the {context['denied_entity']} was {context['denied_state']}"
        asserted_phrase = f"the {context['asserted_entity']} was {context['asserted_state']}"
        if denied_phrase not in premise or asserted_phrase not in premise:
            return "failed:negation_scope_fact_missing"
        if label == "entailment" and asserted_phrase not in hypothesis:
            return "failed:negation_scope_wrong_entailment"
        if label == "contradiction" and denied_phrase not in hypothesis:
            return "failed:negation_scope_wrong_contradiction"
        return "passed"

    if subtype == "not_all":
        total = context["total"]
        incomplete = context["incomplete"]
        if not 0 < incomplete < total:
            return "failed:negation_not_all_invalid_counts"
        if label == "entailment" and "not every" not in hypothesis and "some" not in hypothesis:
            return "failed:negation_not_all_weak_entailment"
        if label == "contradiction" and not hypothesis.startswith("All "):
            return "failed:negation_not_all_wrong_contradiction"
        return "passed"

    if subtype == "none":
        if "none of" not in premise:
            return "failed:negation_none_missing_none"
        if label == "entailment" and not hypothesis.startswith("No "):
            return "failed:negation_none_wrong_entailment"
        if label == "contradiction" and "at least one" not in hypothesis:
            return "failed:negation_none_wrong_contradiction"
        return "passed"

    return "failed:unknown_negation_subtype"


def validate_numeric(candidate: dict) -> str:
    context = candidate["validation"]
    example = candidate["example"]
    label = example["label"]
    hypothesis = example["hypothesis"].lower()
    subtype = context["subtype"]

    if subtype == "exact_count":
        if label == "entailment" and str(context["quantity"]) not in hypothesis:
            return "failed:numeric_exact_missing_quantity"
        if label == "contradiction" and str(context["mismatch"]) not in hypothesis:
            return "failed:numeric_exact_wrong_contradiction"
        return "passed"

    if subtype == "comparison":
        if context["count_a"] == context["count_b"]:
            return "failed:numeric_comparison_equal_counts"
        if label in {"entailment", "contradiction"} and "more" not in hypothesis:
            return "failed:numeric_comparison_missing_relation"
        return "passed"

    if subtype == "bounds":
        quantity = context["quantity"]
        numbers = [int(match) for match in re.findall(r"\d+", hypothesis)]
        if label in {"entailment", "contradiction"} and not numbers:
            return "failed:numeric_bounds_missing_number"
        if label == "entailment":
            bound = numbers[0]
            if "at least" in hypothesis and bound > quantity:
                return "failed:numeric_bounds_bad_lower_bound"
            if "at most" in hypothesis and bound < quantity:
                return "failed:numeric_bounds_bad_upper_bound"
        if label == "contradiction":
            bound = numbers[0]
            if "at least" in hypothesis and bound <= quantity:
                return "failed:numeric_bounds_weak_contradiction"
            if "at most" in hypothesis and bound >= quantity:
                return "failed:numeric_bounds_weak_contradiction"
        return "passed"

    if subtype == "parts_total":
        if label == "entailment" and str(context["total"]) not in hypothesis:
            return "failed:numeric_parts_total_wrong_total"
        if label == "contradiction" and str(context["mismatch"]) not in hypothesis:
            return "failed:numeric_parts_total_wrong_contradiction"
        return "passed"

    return "failed:unknown_numeric_subtype"

def validate_temporal(candidate: dict) -> str:
    context = candidate["validation"]
    example = candidate["example"]
    label = example["label"]
    hypothesis = example["hypothesis"].lower()
    subtype = context["subtype"]

    if subtype == "before_after":
        if not context["date_one"] < context["date_two"]:
            return "failed:temporal_before_after_invalid_order"
        if label == "entailment" and "before" not in hypothesis:
            return "failed:temporal_before_after_missing_before"
        if label == "contradiction" and "after" not in hypothesis:
            return "failed:temporal_before_after_wrong_contradiction"
        return "passed"

    if subtype == "same_day":
        if label == "entailment" and "same day" not in hypothesis:
            return "failed:temporal_same_day_missing_same_day"
        if label == "contradiction" and "different days" not in hypothesis:
            return "failed:temporal_same_day_wrong_contradiction"
        return "passed"

    if subtype == "duration":
        if label == "entailment" and "more than one day" not in hypothesis:
            return "failed:temporal_duration_weak_entailment"
        if label == "contradiction":
            numbers = [int(match) for match in re.findall(r"\d+", hypothesis)]
            if not numbers or numbers[0] == context["duration_days"]:
                return "failed:temporal_duration_missing_bad_duration"
        return "passed"

    if subtype == "sequence":
        if not (context["submitted_dt"] < context["approved_dt"] < context["printed_dt"]):
            return "failed:temporal_sequence_invalid_order"
        if label == "entailment" and "after" not in hypothesis:
            return "failed:temporal_sequence_weak_entailment"
        if label == "contradiction" and "before" not in hypothesis:
            return "failed:temporal_sequence_wrong_contradiction"
        return "passed"

    if subtype == "day_relation":
        if not context["first_date"] < context["later_date"]:
            return "failed:temporal_day_relation_invalid_dates"
        if label == "entailment" and "later" not in hypothesis:
            return "failed:temporal_day_relation_weak_entailment"
        if label == "contradiction" and "same day" not in hypothesis:
            return "failed:temporal_day_relation_wrong_contradiction"
        return "passed"

    return "failed:unknown_temporal_subtype"


def validate_long_reasoning(candidate: dict) -> str:
    context = candidate["validation"]
    example = candidate["example"]
    label = example["label"]
    premise = example["premise"]
    hypothesis = example["hypothesis"]
    subtype = context["subtype"]

    if premise.count(". ") < 2:
        return "failed:long_reasoning_requires_multiple_sentences"
    if len(hypothesis.split()) > 12:
        return "failed:long_reasoning_hypothesis_too_long"

    if subtype == "transfer":
        if label == "entailment" and context["moved_place"] not in hypothesis:
            return "failed:long_reasoning_transfer_wrong_entailment"
        if label == "contradiction" and "damaged" not in hypothesis.lower():
            return "failed:long_reasoning_transfer_weak_contradiction"
        return "passed"

    if subtype == "access":
        if label == "entailment" and context["person_c"] not in hypothesis:
            return "failed:long_reasoning_access_wrong_entailment"
        if label == "contradiction" and context["person_a"] not in hypothesis:
            return "failed:long_reasoning_access_wrong_contradiction"
        return "passed"

    if subtype == "assignment":
        if label == "entailment" and context["drafter"] not in hypothesis:
            return "failed:long_reasoning_assignment_wrong_entailment"
        if label == "contradiction" and context["reviewer"] not in hypothesis:
            return "failed:long_reasoning_assignment_wrong_contradiction"
        return "passed"

    if subtype == "incident":
        if label == "entailment" and "after" not in hypothesis.lower():
            return "failed:long_reasoning_incident_weak_entailment"
        if label == "contradiction" and "before" not in hypothesis.lower():
            return "failed:long_reasoning_incident_wrong_contradiction"
        return "passed"

    return "failed:unknown_long_reasoning_subtype"


def validate_generated_candidate(candidate: dict) -> str:
    example = candidate["example"]
    generic_status = validate_generic_example(example)
    if generic_status != "passed":
        return generic_status

    category_key = candidate["validation"]["category_key"]
    if category_key == "negation":
        return validate_negation(candidate)
    if category_key == "numeric":
        return validate_numeric(candidate)
    if category_key == "temporal":
        return validate_temporal(candidate)
    if category_key == "long_reasoning":
        return validate_long_reasoning(candidate)
    return "failed:unknown_category"


def finalize_candidate(candidate: dict, validation_status: str) -> dict:
    example = dict(candidate["example"])
    example["validation_status"] = validation_status
    return example


CATEGORY_BUILDERS = {
    "negation": [
        build_negation_direct,
        build_negation_scope,
        build_negation_not_all,
        build_negation_none,
    ],
    "numeric": [
        build_numeric_exact_count,
        build_numeric_comparison,
        build_numeric_bounds,
        build_numeric_parts_total,
    ],
    "temporal": [
        build_temporal_before_after,
        build_temporal_same_day,
        build_temporal_duration,
        build_temporal_sequence,
        build_temporal_day_relation,
    ],
    "long_reasoning": [
        build_long_reasoning_transfer,
        build_long_reasoning_access,
        build_long_reasoning_assignment,
        build_long_reasoning_incident,
    ],
}


def generate_category_examples(
    *,
    category_key: str,
    count: int,
    base_seed: int,
) -> tuple[list[dict], dict[str, int]]:
    builders = CATEGORY_BUILDERS[category_key]
    examples: list[dict] = []
    failure_reasons: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str, str]] = set()
    attempts = 0
    max_attempts = max(count * 20, 20)

    while len(examples) < count and attempts < max_attempts:
        builder_index = attempts % len(builders)
        label_cycle = ((attempts // len(builders)) + builder_index) % 3
        example_seed = base_seed + CATEGORY_SEED_OFFSETS[category_key] + attempts
        rng = random.Random(example_seed)
        candidate = builders[builder_index](label_cycle, rng, example_seed)
        validation_status = validate_generated_candidate(candidate)
        finalized = finalize_candidate(candidate, validation_status)
        attempts += 1

        if validation_status != "passed":
            failure_reasons[validation_status] += 1
            continue

        fingerprint = (
            finalized["premise"],
            finalized["hypothesis"],
            finalized["label"],
        )
        if fingerprint in seen_pairs:
            failure_reasons["failed:duplicate_example"] += 1
            continue

        seen_pairs.add(fingerprint)
        examples.append(finalized)

    if len(examples) != count:
        raise RuntimeError(
            f"Requested {count} {category_key} examples but generated {len(examples)} unique valid examples after {attempts} attempts"
        )

    return examples, dict(failure_reasons)


def derive_rejected_output_path(output_path: Path, explicit_rejected_output_path: str | None) -> Path:
    if explicit_rejected_output_path:
        return resolve_path(explicit_rejected_output_path)
    return output_path.with_name(f"rejected_{output_path.name}")


def derive_validation_report_path(output_path: Path, explicit_validation_report_path: str | None) -> Path:
    if explicit_validation_report_path:
        return resolve_path(explicit_validation_report_path)
    return output_path.with_name(f"{output_path.stem}_validation_report.json")


def ensure_safe_output_paths(*candidate_paths: Path) -> None:
    unique_paths: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in unique_paths:
            raise ValueError("Output, rejected, summary, and validation report paths must be different files")
        unique_paths.add(candidate)
        if candidate in PROTECTED_SOURCE_FILES:
            raise ValueError(f"Refusing to write to protected source file: {format_output_path(candidate)}")


def generate_targeted_dataset(args: argparse.Namespace) -> tuple[list[dict], dict]:
    output_path = resolve_path(args.output)
    summary_path = derive_summary_path(output_path, args.summary_output)
    rejected_output_path = derive_rejected_output_path(output_path, args.rejected_output)
    validation_report_path = derive_validation_report_path(output_path, args.validation_report_output)
    ensure_safe_output_paths(output_path, summary_path, rejected_output_path, validation_report_path)

    requested_counts = {
        "negation": args.negation,
        "numeric": args.numeric,
        "temporal": args.temporal,
        "long_reasoning": args.long_reasoning,
    }

    generated_examples: list[dict] = []
    generation_discarded_by_category: dict[str, dict[str, int]] = {CATEGORY_LABELS[key]: {} for key in requested_counts}

    for category_key, count in requested_counts.items():
        if count == 0:
            continue
        category_examples, failure_reasons = generate_category_examples(
            category_key=category_key,
            count=count,
            base_seed=args.seed,
        )
        generated_examples.extend(category_examples)
        generation_discarded_by_category[CATEGORY_LABELS[category_key]] = failure_reasons

    accepted_examples, rejected_examples, validation_report = validate_synthetic_nli_examples(generated_examples)
    accepted_counts = Counter(example["category"] for example in accepted_examples)
    counts_by_category: dict[str, int] = {
        CATEGORY_LABELS[key]: accepted_counts.get(CATEGORY_LABELS[key], 0) for key in requested_counts
    }

    validation_report["accepted_output_path"] = format_output_path(output_path)
    validation_report["rejected_output_path"] = format_output_path(rejected_output_path)
    validation_report["validation_report_path"] = format_output_path(validation_report_path)

    write_json(output_path, accepted_examples)
    write_json(rejected_output_path, rejected_examples)
    write_json(validation_report_path, validation_report)

    summary = {
        "total_generated": len(generated_examples),
        "total_accepted": len(accepted_examples),
        "total_rejected": len(rejected_examples),
        "counts_by_category": counts_by_category,
        "requested_counts": requested_counts,
        "seed": args.seed,
        "output_path": format_output_path(output_path),
        "rejected_output_path": format_output_path(rejected_output_path),
        "validation_report_path": format_output_path(validation_report_path),
        "summary_path": format_output_path(summary_path),
        "generation_discarded_by_category": generation_discarded_by_category,
        "config_path": args.config,
    }
    write_json(summary_path, summary)
    return accepted_examples, summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _, summary = generate_targeted_dataset(args)
    print(f"Saved {summary['total_accepted']} accepted synthetic examples to {summary['output_path']}")
    print(f"Saved {summary['total_rejected']} rejected synthetic examples to {summary['rejected_output_path']}")
    print(f"Saved validation report to {summary['validation_report_path']}")
    print(f"Saved generation summary to {summary['summary_path']}")
    for category_name, count in summary["counts_by_category"].items():
        print(f"- {category_name}: {count}")

if __name__ == "__main__":
    main()

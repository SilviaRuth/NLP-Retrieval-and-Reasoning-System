from __future__ import annotations

import random
import re


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
NEGATION_TERMS = {"no", "not", "never", "none", "nobody", "nothing", "without", "n't"}
PARAPHRASE_MAP = {
    "big": "large",
    "small": "little",
    "shows": "demonstrates",
    "show": "demonstrate",
    "because": "since",
    "buy": "purchase",
    "car": "vehicle",
    "kids": "children",
}


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def lexical_overlap_ratio(text_a: str, text_b: str) -> float:
    tokens_a = set(simple_tokenize(text_a))
    tokens_b = set(simple_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def contains_negation(text: str) -> bool:
    return any(token in NEGATION_TERMS for token in simple_tokenize(text))


def typo_noise(text: str, rate: float = 0.1, seed: int = 42) -> str:
    rng = random.Random(seed)
    chars = list(text)
    swaps = max(1, int(len(chars) * rate))
    for _ in range(swaps):
        if len(chars) < 2:
            break
        index = rng.randrange(0, len(chars) - 1)
        if chars[index].isspace() or chars[index + 1].isspace():
            continue
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
    return "".join(chars)


def paraphrase_text(text: str) -> str:
    tokens = simple_tokenize(text)
    rewritten = [PARAPHRASE_MAP.get(token, token) for token in tokens]
    return " ".join(rewritten)


def shuffle_words(text: str, seed: int = 42) -> str:
    rng = random.Random(seed)
    tokens = simple_tokenize(text)
    if len(tokens) < 4:
        return text
    middle = tokens[1:-1]
    rng.shuffle(middle)
    return " ".join([tokens[0], *middle, tokens[-1]])

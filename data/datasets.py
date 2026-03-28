from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from utils.text import simple_tokenize


LABEL_ALIASES = {
    "entails": "entailment",
    "entailment": "entailment",
    "neutral": "neutral",
    "contradicts": "contradiction",
    "contradiction": "contradiction",
}


@dataclass(frozen=True)
class NLIExample:
    premise: str
    hypothesis: str
    label: str


class Vocabulary:
    def __init__(self, special_tokens: Sequence[str] | None = None) -> None:
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        for token in special_tokens or ("<pad>", "<unk>", "[CLS]", "[SEP]"):
            self.add_word(token)

    def add_word(self, word: str) -> int:
        if word not in self.word2idx:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]

    def lookup(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx["<unk>"])

    def __len__(self) -> int:
        return len(self.idx2word)


def normalize_label(label: str) -> str:
    lowered = label.strip().lower()
    return LABEL_ALIASES.get(lowered, lowered)


def _coerce_row(item: dict) -> NLIExample:
    return NLIExample(
        premise=str(item["premise"]),
        hypothesis=str(item["hypothesis"]),
        label=normalize_label(str(item["label"])),
    )


def load_nli_dataset(path: str | Path) -> list[NLIExample]:
    path = Path(path)
    # Accept UTF-8 files with or without a BOM so datasets produced by
    # PowerShell or spreadsheet export tooling still load cleanly.
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return [_coerce_row(item) for item in payload]

    required = {"premise", "hypothesis", "label"}
    if isinstance(payload, dict) and required.issubset(payload.keys()):
        premise = payload["premise"]
        hypothesis = payload["hypothesis"]
        label = payload["label"]
        if isinstance(premise, dict) and isinstance(hypothesis, dict) and isinstance(label, dict):
            rows: list[NLIExample] = []
            # Support the original notebook export format where each field is stored
            # as a column-oriented dict keyed by row index.
            for key in premise:
                rows.append(
                    NLIExample(
                        premise=str(premise[key]),
                        hypothesis=str(hypothesis[key]),
                        label=normalize_label(str(label[key])),
                    )
                )
            return rows

    raise ValueError(f"Unsupported dataset format in {path}")


def infer_label_map(examples: Iterable[NLIExample]) -> dict[str, int]:
    labels = {example.label for example in examples}
    preferred_order = ["entailment", "contradiction", "neutral"]
    ordered = [label for label in preferred_order if label in labels]
    ordered.extend(sorted(labels - set(ordered)))
    return {label: index for index, label in enumerate(ordered)}


def build_vocab(examples: Iterable[NLIExample], min_freq: int = 2) -> Vocabulary:
    counter: Counter[str] = Counter()
    for example in examples:
        counter.update(simple_tokenize(example.premise.lower()))
        counter.update(simple_tokenize(example.hypothesis.lower()))

    vocab = Vocabulary()
    for token, count in counter.items():
        if count >= min_freq:
            vocab.add_word(token)
    return vocab


class TokenPairDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[NLIExample],
        vocab: Vocabulary,
        label_to_id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        self.examples = list(examples)
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.max_length = max_length
        self.pad_idx = vocab.word2idx["<pad>"]

    def __len__(self) -> int:
        return len(self.examples)

    def _encode_pair(self, premise: str, hypothesis: str) -> tuple[list[int], list[int]]:
        # Baseline models do not use a pretrained tokenizer, so we manually mimic
        # the standard pair layout: [CLS] premise [SEP] hypothesis [SEP].
        tokens = ["[CLS]"]
        tokens.extend(simple_tokenize(premise.lower()))
        tokens.append("[SEP]")
        tokens.extend(simple_tokenize(hypothesis.lower()))
        tokens.append("[SEP]")

        token_ids = [self.vocab.lookup(token) for token in tokens[: self.max_length]]
        attention_mask = [1] * len(token_ids)

        if len(token_ids) < self.max_length:
            pad_length = self.max_length - len(token_ids)
            token_ids.extend([self.pad_idx] * pad_length)
            attention_mask.extend([0] * pad_length)

        return token_ids, attention_mask

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        input_ids, attention_mask = self._encode_pair(example.premise, example.hypothesis)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[example.label], dtype=torch.long),
        }


class TransformerNLIDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[NLIExample],
        tokenizer,
        label_to_id: dict[str, int] | None = None,
        max_length: int = 256,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        # Hugging Face tokenizers already handle pair packing, truncation, and masks.
        encoded = self.tokenizer(
            example.premise,
            example.hypothesis,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.squeeze(0) for key, value in encoded.items()}
        if self.label_to_id is not None:
            batch["labels"] = torch.tensor(self.label_to_id[example.label], dtype=torch.long)
        return batch

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

from utils.io import ensure_dir, write_json

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForSequenceClassification = None
    AutoTokenizer = None


@dataclass
class BertNLIConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    local_files_only: bool = False


def _require_transformers() -> None:
    if AutoModelForSequenceClassification is None or AutoTokenizer is None:
        raise ImportError("transformers is required for BERT training and inference")


def build_tokenizer(model_name: str, local_files_only: bool = False):
    _require_transformers()
    return AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)


class BertNLIClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_labels: int,
        label_to_id: dict[str, int],
        id_to_label: dict[int, str],
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        _require_transformers()
        self.model_name = model_name
        self.label_to_id = label_to_id
        self.id_to_label = id_to_label
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            label2id=label_to_id,
            id2label=id_to_label,
            local_files_only=local_files_only,
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def save_checkpoint(self, output_dir: str | Path, tokenizer=None, config: BertNLIConfig | None = None) -> None:
        output_dir = ensure_dir(output_dir)
        self.model.save_pretrained(output_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        if config is not None:
            write_json(Path(output_dir) / "training_config.json", asdict(config))

    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir: str | Path) -> "BertNLIClassifier":
        checkpoint_dir = Path(checkpoint_dir)
        _require_transformers()
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True)
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.model_name = str(checkpoint_dir)
        instance.label_to_id = dict(model.config.label2id)
        instance.id_to_label = {int(key): value for key, value in model.config.id2label.items()}
        instance.model = model
        return instance

    def predict_proba(
        self,
        tokenizer,
        premises: list[str],
        hypotheses: list[str],
        max_length: int = 256,
        batch_size: int = 16,
        device: torch.device | None = None,
    ) -> list[list[float]]:
        device = device or next(self.parameters()).device
        self.eval()
        probabilities: list[list[float]] = []
        with torch.no_grad():
            for start in range(0, len(premises), batch_size):
                batch_premises = premises[start : start + batch_size]
                batch_hypotheses = hypotheses[start : start + batch_size]
                encoded = tokenizer(
                    batch_premises,
                    batch_hypotheses,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = self.model(**encoded).logits
                probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
        return probabilities

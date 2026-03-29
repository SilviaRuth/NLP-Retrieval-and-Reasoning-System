from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from data import NLIExample, infer_label_map, load_nli_dataset
from evaluation.metrics import classification_metrics
from evaluation.robustness import evaluate_robustness
from models import BertNLIClassifier, build_tokenizer
from utils import read_json


def load_checkpoint_runtime(checkpoint_dir: str | Path):
    checkpoint_dir = Path(checkpoint_dir)
    model = BertNLIClassifier.load_from_checkpoint(checkpoint_dir)
    tokenizer = build_tokenizer(checkpoint_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    training_config = read_json(checkpoint_dir / "training_config.json") if (checkpoint_dir / "training_config.json").exists() else {}
    max_length = int(training_config.get("max_length", 256))
    batch_size = int(training_config.get("batch_size", 8))
    return model, tokenizer, device, training_config, max_length, batch_size


def predict_checkpoint(
    checkpoint_dir: str | Path,
    examples: list[NLIExample],
    max_length: int | None = None,
    batch_size: int | None = None,
) -> tuple[list[int], list[int], dict[str, int], dict[int, str], float]:
    model, tokenizer, device, training_config, default_max_length, default_batch_size = load_checkpoint_runtime(checkpoint_dir)
    max_length = max_length or default_max_length
    batch_size = batch_size or default_batch_size
    label_to_id = dict(model.label_to_id)
    id_to_label = {int(index): label for index, label in model.id_to_label.items()}

    probabilities: list[list[float]] = []
    total_loss = 0.0
    batches = 0
    gold = [label_to_id[example.label] for example in examples]

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start : start + batch_size]
            encoded = tokenizer(
                [example.premise for example in batch_examples],
                [example.hypothesis for example in batch_examples],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = torch.tensor([label_to_id[example.label] for example in batch_examples], dtype=torch.long, device=device)
            logits = model.model(**encoded).logits
            probabilities.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += F.cross_entropy(logits, labels).item()
            batches += 1

    predicted = [max(range(len(probability)), key=probability.__getitem__) for probability in probabilities]
    average_loss = total_loss / batches if batches else 0.0
    return gold, predicted, label_to_id, id_to_label, average_loss


def evaluate_checkpoint_dataset(
    checkpoint_dir: str | Path,
    dataset_path: str | Path,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> dict:
    examples = load_nli_dataset(dataset_path)
    gold, predicted, label_to_id, _, average_loss = predict_checkpoint(
        checkpoint_dir=checkpoint_dir,
        examples=examples,
        max_length=max_length,
        batch_size=batch_size,
    )
    metrics = classification_metrics(gold, predicted, labels=label_to_id.values())
    metrics["loss"] = average_loss
    return metrics


def make_checkpoint_predict_fn(
    checkpoint_dir: str | Path,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> tuple[Callable[[list[NLIExample]], list[int]], dict[str, int]]:
    model, tokenizer, device, _, default_max_length, default_batch_size = load_checkpoint_runtime(checkpoint_dir)
    max_length = max_length or default_max_length
    batch_size = batch_size or default_batch_size
    label_to_id = dict(model.label_to_id)

    def predict(examples: list[NLIExample]) -> list[int]:
        probabilities = model.predict_proba(
            tokenizer=tokenizer,
            premises=[example.premise for example in examples],
            hypotheses=[example.hypothesis for example in examples],
            max_length=max_length,
            batch_size=batch_size,
            device=device,
        )
        return [max(range(len(probability)), key=probability.__getitem__) for probability in probabilities]

    return predict, label_to_id


def evaluate_checkpoint_robustness(
    checkpoint_dir: str | Path,
    dataset_path: str | Path,
    max_length: int | None = None,
    batch_size: int | None = None,
) -> dict:
    examples = load_nli_dataset(dataset_path)
    predict_fn, label_to_id = make_checkpoint_predict_fn(
        checkpoint_dir=checkpoint_dir,
        max_length=max_length,
        batch_size=batch_size,
    )
    return evaluate_robustness(examples, predict_fn, label_to_id)


def checkpoint_supports_inference(run_dir: str | Path) -> bool:
    run_dir = Path(run_dir)
    return (run_dir / "model.safetensors").exists() and (run_dir / "config.json").exists()


def infer_label_to_id_from_run(run_dir: str | Path) -> dict[str, int]:
    run_dir = Path(run_dir)
    label_map_path = run_dir / "label_map.json"
    if label_map_path.exists():
        return read_json(label_map_path)
    if checkpoint_supports_inference(run_dir):
        model, _, _, _, _, _ = load_checkpoint_runtime(run_dir)
        return dict(model.label_to_id)
    return infer_label_map(load_nli_dataset(Path("data") / "validation.json"))

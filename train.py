from __future__ import annotations

import argparse
import copy
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import (
    TokenPairDataset,
    TransformerNLIDataset,
    build_vocab,
    infer_label_map,
    load_nli_dataset,
)
from evaluation import analyze_errors, classification_metrics, evaluate_robustness
from experiments import ExperimentTracker
from models import BertNLIClassifier, BertNLIConfig, BiLSTMNLIClassifier, CNNNLIClassifier, build_tokenizer
from utils import ensure_dir, set_seed, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an NLI classification model.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--test-path")
    parser.add_argument("--output-dir", default="outputs/run")
    parser.add_argument("--model-type", choices=["bert", "bilstm", "lstm", "cnn"], default="bert")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-freq", type=int, default=2)
    return parser.parse_args()


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def forward_with_loss(model, batch: dict[str, torch.Tensor], criterion):
    labels = batch["labels"]
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    outputs = model(**inputs)
    if hasattr(outputs, "logits"):
        logits = outputs.logits
        loss = outputs.loss if outputs.loss is not None else criterion(logits, labels)
    else:
        logits = outputs
        loss = criterion(logits, labels)
    return loss, logits, labels


def train_epoch(model, dataloader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        loss, _, _ = forward_with_loss(model, batch, criterion)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def evaluate_model(model, dataloader, criterion, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    total_loss = 0.0
    all_gold: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = move_batch_to_device(batch, device)
            loss, logits, labels = forward_with_loss(model, batch, criterion)
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            all_gold.extend(labels.cpu().tolist())
            all_pred.extend(predictions.cpu().tolist())
    return total_loss / max(len(dataloader), 1), all_gold, all_pred


def make_predict_fn(
    model,
    tokenizer,
    vocab,
    label_to_id,
    max_length: int,
    batch_size: int,
    device: torch.device,
):
    def predict(examples):
        if tokenizer is not None:
            dataset = TransformerNLIDataset(examples, tokenizer, label_to_id=label_to_id, max_length=max_length)
        else:
            dataset = TokenPairDataset(examples, vocab, label_to_id=label_to_id, max_length=max_length)
        dataloader = DataLoader(dataset, batch_size=batch_size)
        _, _, predictions = evaluate_model(model, dataloader, nn.CrossEntropyLoss(), device)
        return predictions

    return predict


def save_baseline_checkpoint(output_dir, model, vocab, label_to_id, args: argparse.Namespace) -> None:
    torch.save(model.state_dict(), output_dir / "model.pt")
    write_json(output_dir / "vocab.json", vocab.word2idx)
    write_json(output_dir / "label_map.json", label_to_id)
    write_json(output_dir / "training_config.json", vars(args))


def main() -> None:
    args = parse_args()
    if args.model_type == "lstm":
        args.model_type = "bilstm"

    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = load_nli_dataset(args.train_path)
    val_examples = load_nli_dataset(args.val_path)
    test_examples = load_nli_dataset(args.test_path) if args.test_path else []
    label_to_id = infer_label_map(train_examples + val_examples + test_examples)
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = None
    vocab = None

    if args.model_type == "bert":
        tokenizer = build_tokenizer(args.model_name, local_files_only=args.local_files_only)
        train_dataset = TransformerNLIDataset(train_examples, tokenizer, label_to_id, args.max_length)
        val_dataset = TransformerNLIDataset(val_examples, tokenizer, label_to_id, args.max_length)
        test_dataset = TransformerNLIDataset(test_examples, tokenizer, label_to_id, args.max_length) if test_examples else None

        config = BertNLIConfig(
            model_name=args.model_name,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            local_files_only=args.local_files_only,
        )
        model = BertNLIClassifier(
            model_name=args.model_name,
            num_labels=len(label_to_id),
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            local_files_only=args.local_files_only,
        )
    else:
        vocab = build_vocab(train_examples, min_freq=args.min_freq)
        train_dataset = TokenPairDataset(train_examples, vocab, label_to_id, args.max_length)
        val_dataset = TokenPairDataset(val_examples, vocab, label_to_id, args.max_length)
        test_dataset = TokenPairDataset(test_examples, vocab, label_to_id, args.max_length) if test_examples else None

        if args.model_type == "bilstm":
            model = BiLSTMNLIClassifier(
                vocab_size=len(vocab),
                pad_idx=vocab.word2idx["<pad>"],
                num_labels=len(label_to_id),
            )
        else:
            model = CNNNLIClassifier(
                vocab_size=len(vocab),
                pad_idx=vocab.word2idx["<pad>"],
                num_labels=len(label_to_id),
            )
        config = vars(args)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size) if test_dataset is not None else None

    best_val_f1 = -1.0
    best_metrics = None
    best_predictions = None
    best_state_dict = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_gold, val_pred = evaluate_model(model, val_loader, criterion, device)
        val_metrics = classification_metrics(val_gold, val_pred, labels=label_to_id.values())
        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_accuracy={val_metrics['accuracy']:.4f} | val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            best_predictions = (val_gold, val_pred)
            best_state_dict = copy.deepcopy(model.state_dict())
            if args.model_type == "bert":
                model.save_checkpoint(output_dir, tokenizer=tokenizer, config=config)
            else:
                save_baseline_checkpoint(output_dir, model, vocab, label_to_id, args)

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    write_json(output_dir / "label_map.json", label_to_id)
    write_json(output_dir / "metrics.json", best_metrics)

    val_gold, val_pred = best_predictions
    error_report = analyze_errors(val_examples, val_gold, val_pred, id_to_label)
    write_json(output_dir / "error_analysis.json", error_report)

    predict_fn = make_predict_fn(
        model=model,
        tokenizer=tokenizer,
        vocab=vocab,
        label_to_id=label_to_id,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )
    robustness_report = evaluate_robustness(val_examples, predict_fn, label_to_id)
    write_json(output_dir / "robustness.json", robustness_report)

    if test_loader is not None:
        test_loss, test_gold, test_pred = evaluate_model(model, test_loader, criterion, device)
        test_metrics = classification_metrics(test_gold, test_pred, labels=label_to_id.values())
        test_metrics["loss"] = test_loss
        write_json(output_dir / "test_metrics.json", test_metrics)

    tracker = ExperimentTracker()
    tracker.log_run(
        run_name=args.model_type,
        config=asdict(config) if isinstance(config, BertNLIConfig) else config,
        metrics=best_metrics,
        artifacts={"output_dir": str(output_dir)},
    )


if __name__ == "__main__":
    main()

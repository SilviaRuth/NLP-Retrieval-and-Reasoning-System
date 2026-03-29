from __future__ import annotations

import argparse
import copy
import math
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
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
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--min-freq", type=int, default=2)
    return parser.parse_args()


def resolve_training_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.learning_rate is None:
        args.learning_rate = 2e-5 if args.model_type == "bert" else 1e-3
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 4 if args.model_type == "bert" else 1
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    if args.warmup_steps is not None and args.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in the range [0.0, 1.0)")
    if args.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    return args


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


def build_linear_warmup_scheduler(optimizer, total_steps: int, warmup_steps: int) -> LambdaLR | None:
    if total_steps <= 0:
        return None

    capped_warmup_steps = min(warmup_steps, total_steps)

    def lr_lambda(current_step: int) -> float:
        if capped_warmup_steps > 0 and current_step < capped_warmup_steps:
            return float(current_step + 1) / float(capped_warmup_steps)

        remaining_steps = max(total_steps - capped_warmup_steps, 1)
        return max(0.0, float(total_steps - (current_step + 1)) / float(remaining_steps))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device: torch.device,
    scheduler: LambdaLR | None = None,
    gradient_accumulation_steps: int = 1,
    max_grad_norm: float | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        batch = move_batch_to_device(batch, device)
        loss, _, _ = forward_with_loss(model, batch, criterion)
        raw_loss = loss
        (loss / gradient_accumulation_steps).backward()

        should_step = step % gradient_accumulation_steps == 0 or step == len(dataloader)
        if should_step:
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        total_loss += raw_loss.item()
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
    args = resolve_training_defaults(args)

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
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            early_stopping_patience=args.early_stopping_patience,
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

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_training_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else math.ceil(total_training_steps * args.warmup_ratio)
    scheduler = build_linear_warmup_scheduler(
        optimizer=optimizer,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps,
    )
    args.warmup_steps = warmup_steps
    if isinstance(config, BertNLIConfig):
        config.warmup_steps = warmup_steps

    best_val_f1 = -1.0
    best_metrics = None
    best_predictions = None
    best_state_dict = None
    epochs_without_improvement = 0

    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(
        f"Training {args.model_type} with lr={args.learning_rate:g}, "
        f"batch_size={args.batch_size}, effective_batch_size={effective_batch_size}, "
        f"warmup_steps={warmup_steps}, total_optimizer_steps={total_training_steps}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scheduler=scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
        )
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
            epochs_without_improvement = 0
            if args.model_type == "bert":
                model.save_checkpoint(output_dir, tokenizer=tokenizer, config=config)
            else:
                save_baseline_checkpoint(output_dir, model, vocab, label_to_id, args)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    f"Early stopping after epoch {epoch}: "
                    f"validation macro F1 did not improve for {epochs_without_improvement} consecutive epochs."
                )
                break

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


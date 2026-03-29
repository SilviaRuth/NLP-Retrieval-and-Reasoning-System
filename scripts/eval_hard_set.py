from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import NLIExample
from evaluation import compute_bucket_metrics, load_hard_set, persist_hard_set_metrics, predict_checkpoint
from evaluation.metrics import classification_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the curated hard set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hard-set-path", default="data/eval/hard_set.json")
    parser.add_argument("--output-json")
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = load_hard_set(args.hard_set_path)
    hard_examples = payload.get("examples", [])
    dataset = [
        NLIExample(
            premise=example["premise"],
            hypothesis=example["hypothesis"],
            label=example["gold_label"],
        )
        for example in hard_examples
    ]

    gold, predicted, label_to_id, _, average_loss = predict_checkpoint(
        checkpoint_dir=args.checkpoint,
        examples=dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    metrics = classification_metrics(gold, predicted, labels=label_to_id.values())
    metrics["loss"] = average_loss
    metrics["count"] = len(dataset)
    metrics["buckets"] = compute_bucket_metrics(hard_examples, gold, predicted, label_to_id)

    output_path = args.output_json or f"{args.checkpoint}/hard_set_metrics.json"
    persist_hard_set_metrics(args.checkpoint, metrics)
    if output_path != f"{args.checkpoint}/hard_set_metrics.json":
        from utils import write_json
        write_json(output_path, metrics)

    print(f"Hard-set accuracy: {metrics['accuracy']:.4f}")
    print(f"Hard-set macro F1: {metrics['macro_f1']:.4f}")
    for bucket, bucket_metrics in metrics["buckets"].items():
        print(f"- {bucket}: count={bucket_metrics['count']} accuracy={bucket_metrics['accuracy']:.4f} macro_f1={bucket_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()

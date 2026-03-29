from __future__ import annotations

from pathlib import Path
from typing import Any

from data import NLIExample
from evaluation.checkpoint_eval import checkpoint_supports_inference, evaluate_checkpoint_dataset, evaluate_checkpoint_robustness, predict_checkpoint
from evaluation.hard_set import compute_bucket_metrics, load_hard_set
from evaluation.metrics import classification_metrics
from utils import read_json, write_json


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return read_json(path)
    return None


def load_run_artifacts(run_dir: str | Path, hard_set_path: str | Path = "data/eval/hard_set.json") -> dict[str, Any]:
    run_dir = Path(run_dir)
    training_config = _load_json_if_exists(run_dir / "training_config.json") or {}
    metrics = _load_json_if_exists(run_dir / "metrics.json")
    test_metrics = _load_json_if_exists(run_dir / "test_metrics.json")
    robustness = _load_json_if_exists(run_dir / "robustness.json")
    hard_set_metrics = _load_json_if_exists(run_dir / "hard_set_metrics.json")
    best_run_summary = _load_json_if_exists(run_dir / "best_run_summary.json")
    run_status = _load_json_if_exists(run_dir / "run_status.json")

    if metrics is None and checkpoint_supports_inference(run_dir):
        metrics = evaluate_checkpoint_dataset(run_dir, Path("data") / "validation.json")
    if test_metrics is None and checkpoint_supports_inference(run_dir):
        test_metrics = evaluate_checkpoint_dataset(run_dir, Path("data") / "test.json")
    if robustness is None and checkpoint_supports_inference(run_dir):
        robustness = evaluate_checkpoint_robustness(run_dir, Path("data") / "validation.json")
    if hard_set_metrics is None and checkpoint_supports_inference(run_dir) and Path(hard_set_path).exists():
        hard_payload = load_hard_set(hard_set_path)
        hard_examples = hard_payload.get("examples", [])
        dataset = [
            NLIExample(
                premise=example["premise"],
                hypothesis=example["hypothesis"],
                label=example["gold_label"],
            )
            for example in hard_examples
        ]
        gold, predicted, label_to_id, _, average_loss = predict_checkpoint(run_dir, dataset)
        hard_set_metrics = classification_metrics(gold, predicted, labels=label_to_id.values())
        hard_set_metrics["loss"] = average_loss
        hard_set_metrics["buckets"] = compute_bucket_metrics(hard_examples, gold, predicted, label_to_id)
        hard_set_metrics["count"] = len(hard_examples)

    return {
        "run_dir": str(run_dir),
        "training_config": training_config,
        "metrics": metrics,
        "test_metrics": test_metrics,
        "robustness": robustness,
        "hard_set_metrics": hard_set_metrics,
        "best_run_summary": best_run_summary,
        "run_status": run_status,
        "artifact_paths": {
            "metrics": str(run_dir / "metrics.json"),
            "test_metrics": str(run_dir / "test_metrics.json"),
            "robustness": str(run_dir / "robustness.json"),
            "hard_set_metrics": str(run_dir / "hard_set_metrics.json"),
            "best_run_summary": str(run_dir / "best_run_summary.json"),
            "run_status": str(run_dir / "run_status.json"),
        },
    }


def compare_run_payloads(
    base: dict[str, Any],
    candidate: dict[str, Any],
    hard_set_tolerance: float = 0.01,
    robustness_tolerance: float = 0.02,
) -> dict[str, Any]:
    comparison = {
        "validation_accuracy_delta": _metric_delta(base.get("metrics"), candidate.get("metrics"), "accuracy"),
        "validation_macro_f1_delta": _metric_delta(base.get("metrics"), candidate.get("metrics"), "macro_f1"),
        "test_accuracy_delta": _metric_delta(base.get("test_metrics"), candidate.get("test_metrics"), "accuracy"),
        "test_macro_f1_delta": _metric_delta(base.get("test_metrics"), candidate.get("test_metrics"), "macro_f1"),
        "hard_set_macro_f1_delta": _metric_delta(base.get("hard_set_metrics"), candidate.get("hard_set_metrics"), "macro_f1"),
        "typo_macro_f1_delta": _nested_metric_delta(base.get("robustness"), candidate.get("robustness"), "typo", "macro_f1"),
        "paraphrase_macro_f1_delta": _nested_metric_delta(base.get("robustness"), candidate.get("robustness"), "paraphrase", "macro_f1"),
        "shuffle_macro_f1_delta": _nested_metric_delta(base.get("robustness"), candidate.get("robustness"), "shuffle", "macro_f1"),
    }

    reasons: list[str] = []
    verdict = "inconclusive"
    val_delta = comparison["validation_macro_f1_delta"]
    hard_delta = comparison["hard_set_macro_f1_delta"]
    typo_delta = comparison["typo_macro_f1_delta"]
    shuffle_delta = comparison["shuffle_macro_f1_delta"]

    if val_delta is None or hard_delta is None or typo_delta is None or shuffle_delta is None:
        reasons.append("Missing one or more required metrics for a promotion decision.")
    else:
        if val_delta <= 0:
            reasons.append("Validation macro F1 did not improve.")
        if hard_delta < -hard_set_tolerance:
            reasons.append("Hard-set macro F1 regressed materially.")
        if typo_delta < -robustness_tolerance:
            reasons.append("Typo robustness regressed materially.")
        if shuffle_delta < -robustness_tolerance:
            reasons.append("Shuffle robustness regressed materially.")

        if not reasons:
            verdict = "improved"
            reasons.append("Validation, hard-set, and robustness gates all passed.")
        else:
            verdict = "regressed" if val_delta is not None and val_delta <= 0 else "inconclusive"

    return {
        "base_run": base.get("run_dir"),
        "candidate_run": candidate.get("run_dir"),
        "comparison": comparison,
        "verdict": verdict,
        "reasons": reasons,
        "thresholds": {
            "hard_set_tolerance": hard_set_tolerance,
            "robustness_tolerance": robustness_tolerance,
        },
    }


def render_markdown_summary(report: dict[str, Any]) -> str:
    lines = [
        "## Run Comparison",
        "",
        f"- Base: `{report['base_run']}`",
        f"- Candidate: `{report['candidate_run']}`",
        f"- Verdict: `{report['verdict']}`",
        "",
        "| Metric | Delta |",
        "| --- | --- |",
    ]
    for metric, value in report["comparison"].items():
        display = "n/a" if value is None else f"{value:+.4f}"
        lines.append(f"| {metric} | {display} |")
    lines.append("")
    lines.append("Reasons:")
    for reason in report["reasons"]:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def persist_hard_set_metrics(run_dir: str | Path, payload: dict[str, Any]) -> None:
    write_json(Path(run_dir) / "hard_set_metrics.json", payload)


def _metric_delta(base: dict[str, Any] | None, candidate: dict[str, Any] | None, key: str) -> float | None:
    if not base or not candidate or key not in base or key not in candidate:
        return None
    return candidate[key] - base[key]


def _nested_metric_delta(
    base: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    section: str,
    key: str,
) -> float | None:
    if not base or not candidate:
        return None
    base_section = base.get(section) or {}
    candidate_section = candidate.get(section) or {}
    if key not in base_section or key not in candidate_section:
        return None
    return candidate_section[key] - base_section[key]

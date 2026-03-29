from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import compare_run_payloads, load_run_artifacts, render_markdown_summary
from utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize one run and optionally compare it with a baseline.")
    parser.add_argument("--run", required=True)
    parser.add_argument("--baseline", default="outputs/bert_nli")
    parser.add_argument("--hard-set-path", default="data/eval/hard_set.json")
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_payload = load_run_artifacts(args.run, hard_set_path=args.hard_set_path)
    baseline_payload = load_run_artifacts(args.baseline, hard_set_path=args.hard_set_path) if args.baseline else None

    verdict = {"verdict": "inconclusive", "reasons": ["No baseline comparison requested."]}
    comparison_report = None
    if baseline_payload is not None:
        comparison_report = compare_run_payloads(baseline_payload, run_payload)
        verdict = {
            "verdict": comparison_report["verdict"],
            "reasons": comparison_report["reasons"],
        }

    summary = {
        "run": args.run,
        "training_config": run_payload.get("training_config"),
        "best_run_summary": run_payload.get("best_run_summary"),
        "metrics": run_payload.get("metrics"),
        "test_metrics": run_payload.get("test_metrics"),
        "robustness": run_payload.get("robustness"),
        "hard_set_metrics": run_payload.get("hard_set_metrics"),
        "artifact_paths": run_payload.get("artifact_paths"),
        "final_verdict": verdict,
        "comparison_report": comparison_report,
    }

    stem = Path(args.run).name
    output_json = args.output_json or f"results/{stem}_summary.json"
    output_markdown = args.output_markdown or f"results/{stem}_summary.md"
    write_json(output_json, summary)

    lines = [
        "## Run Summary",
        "",
        f"- Run: `{args.run}`",
        f"- Verdict: `{verdict['verdict']}`",
    ]
    best_run_summary = run_payload.get("best_run_summary") or {}
    if best_run_summary.get("best_epoch") is not None:
        lines.append(f"- Best epoch: `{best_run_summary['best_epoch']}`")
    metrics = run_payload.get("metrics") or {}
    if metrics:
        lines.append(f"- Validation accuracy: `{metrics.get('accuracy', 0.0):.4f}`")
        lines.append(f"- Validation macro F1: `{metrics.get('macro_f1', 0.0):.4f}`")
    test_metrics = run_payload.get("test_metrics") or {}
    if test_metrics:
        lines.append(f"- Test accuracy: `{test_metrics.get('accuracy', 0.0):.4f}`")
        lines.append(f"- Test macro F1: `{test_metrics.get('macro_f1', 0.0):.4f}`")
    hard_set_metrics = run_payload.get("hard_set_metrics") or {}
    if hard_set_metrics:
        lines.append(f"- Hard-set macro F1: `{hard_set_metrics.get('macro_f1', 0.0):.4f}`")
    lines.append("")
    lines.append("Reasons:")
    for reason in verdict["reasons"]:
        lines.append(f"- {reason}")
    if comparison_report is not None:
        lines.append("")
        lines.append(render_markdown_summary(comparison_report))
    Path(output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(output_markdown).write_text("\n".join(lines), encoding="utf-8")

    print(f"Run: {args.run}")
    print(f"Verdict: {verdict['verdict']}")
    for reason in verdict["reasons"]:
        print(f"- {reason}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_markdown}")


if __name__ == "__main__":
    main()

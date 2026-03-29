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
    parser = argparse.ArgumentParser(description="Compare two experiment runs and apply promotion gates.")
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--hard-set-path", default="data/eval/hard_set.json")
    parser.add_argument("--hard-set-tolerance", type=float, default=0.01)
    parser.add_argument("--robustness-tolerance", type=float, default=0.02)
    parser.add_argument("--output-json")
    parser.add_argument("--output-markdown")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_payload = load_run_artifacts(args.base, hard_set_path=args.hard_set_path)
    candidate_payload = load_run_artifacts(args.candidate, hard_set_path=args.hard_set_path)
    report = compare_run_payloads(
        base=base_payload,
        candidate=candidate_payload,
        hard_set_tolerance=args.hard_set_tolerance,
        robustness_tolerance=args.robustness_tolerance,
    )

    default_stem = f"compare_{Path(args.base).name}_vs_{Path(args.candidate).name}"
    output_json = args.output_json or f"results/{default_stem}.json"
    output_markdown = args.output_markdown or f"results/{default_stem}.md"
    markdown = render_markdown_summary(report)
    write_json(output_json, report)
    Path(output_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(output_markdown).write_text(markdown, encoding="utf-8")

    print(f"Verdict: {report['verdict']}")
    for metric, value in report["comparison"].items():
        display = "n/a" if value is None else f"{value:+.4f}"
        print(f"- {metric}: {display}")
    print("Reasons:")
    for reason in report["reasons"]:
        print(f"- {reason}")
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_markdown}")


if __name__ == "__main__":
    main()

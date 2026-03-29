from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from evaluation import build_hard_set, find_latest_error_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated hard-set evaluation file from error analysis.")
    parser.add_argument("--error-analysis-path")
    parser.add_argument("--source-path", default="data/validation.json")
    parser.add_argument("--source-split", default="validation")
    parser.add_argument("--output-path", default="data/eval/hard_set.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_analysis_path = Path(args.error_analysis_path) if args.error_analysis_path else find_latest_error_analysis()
    payload = build_hard_set(
        error_analysis_path=error_analysis_path,
        source_path=args.source_path,
        output_path=args.output_path,
        source_split=args.source_split,
    )
    print(f"Built hard set at {args.output_path}")
    print(f"Source error analysis: {error_analysis_path}")
    print(f"Examples: {payload['num_examples']}")
    for bucket, count in payload["bucket_counts"].items():
        print(f"- {bucket}: {count}")


if __name__ == "__main__":
    main()


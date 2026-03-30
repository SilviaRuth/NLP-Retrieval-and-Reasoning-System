from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import read_json, write_json
from utils.synthetic_nli_validation import validate_synthetic_nli_examples


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def derive_rejected_output_path(output_path: Path, explicit_rejected_output_path: str | None) -> Path:
    if explicit_rejected_output_path:
        return resolve_path(explicit_rejected_output_path)
    return output_path.with_name(f"rejected_{output_path.name}")


def derive_report_output_path(output_path: Path, explicit_report_output_path: str | None) -> Path:
    if explicit_report_output_path:
        return resolve_path(explicit_report_output_path)
    return output_path.with_name(f"{output_path.stem}_validation_report.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and filter synthetic NLI examples.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected-output")
    parser.add_argument("--report-output")
    return parser.parse_args()


def load_examples(path: Path) -> list[dict]:
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("examples"), list):
        return payload["examples"]
    raise ValueError(f"Unsupported synthetic dataset format in {path}")


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    rejected_output_path = derive_rejected_output_path(output_path, args.rejected_output)
    report_output_path = derive_report_output_path(output_path, args.report_output)

    examples = load_examples(input_path)
    accepted_examples, rejected_examples, report = validate_synthetic_nli_examples(examples)
    report["input_path"] = str(input_path)
    report["accepted_output_path"] = str(output_path)
    report["rejected_output_path"] = str(rejected_output_path)
    report["validation_report_path"] = str(report_output_path)

    write_json(output_path, accepted_examples)
    write_json(rejected_output_path, rejected_examples)
    write_json(report_output_path, report)

    print(f"Accepted: {report['total_accepted']}")
    print(f"Rejected: {report['total_rejected']}")
    print(f"Report: {report_output_path}")


if __name__ == "__main__":
    main()

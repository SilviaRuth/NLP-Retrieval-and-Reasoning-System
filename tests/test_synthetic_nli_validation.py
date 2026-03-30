import importlib.util
import tempfile
import unittest
from pathlib import Path

from utils.synthetic_nli_validation import validate_synthetic_nli_examples


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate_targeted_nli_data.py"


def load_generator_module():
    spec = importlib.util.spec_from_file_location("targeted_nli_generator_with_validation", GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load generator module from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SyntheticNLIValidationTests(unittest.TestCase):
    def test_validator_rejects_bad_examples_and_reports_reasons(self) -> None:
        examples = [
            {
                "premise": "The inspector said the alarm panel was not active in the east lab.",
                "hypothesis": "The alarm panel was not active.",
                "label": "entailment",
                "category": "negation",
            },
            {
                "premise": "The inspector said the alarm panel was not active in the east lab.",
                "hypothesis": "The alarm panel was not active.",
                "label": "entailment",
                "category": "negation",
            },
            {
                "premise": "The count was {value}.",
                "hypothesis": "There were 3 boxes on the shelf.",
                "label": "entailment",
                "category": "numeric contradiction",
            },
            {
                "premise": "The review covered three forms in the office.",
                "hypothesis": "The review changed format.",
                "label": "neutral",
                "category": "temporal/date reasoning",
            },
            {
                "premise": "Mina filed the memo.",
                "hypothesis": "Mina filed it.",
                "label": "entailment",
                "category": "long-premise short-hypothesis reasoning",
            },
        ]

        accepted, rejected, report = validate_synthetic_nli_examples(examples)

        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 4)
        self.assertEqual(report["total_generated"], 5)
        self.assertEqual(report["total_accepted"], 1)
        self.assertEqual(report["total_rejected"], 4)
        self.assertIn("duplicate_exact", report["rejection_reasons_by_count"])
        self.assertIn("template_artifact", report["rejection_reasons_by_count"])
        self.assertIn("temporal_missing_temporal_cue", report["rejection_reasons_by_count"])
        self.assertIn("long_reasoning_premise_too_short", report["rejection_reasons_by_count"])

    def test_generator_writes_rejected_examples_and_validation_report(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            output_path = temp_root / "accepted.json"
            rejected_path = temp_root / "rejected.json"
            report_path = temp_root / "validation_report.json"
            summary_path = temp_root / "summary.json"

            args = module.parse_args(
                [
                    "--negation",
                    "4",
                    "--numeric",
                    "4",
                    "--seed",
                    "5",
                    "--output",
                    str(output_path),
                    "--rejected-output",
                    str(rejected_path),
                    "--validation-report-output",
                    str(report_path),
                    "--summary-output",
                    str(summary_path),
                ]
            )
            accepted_examples, summary = module.generate_targeted_dataset(args)

            self.assertTrue(output_path.exists())
            self.assertTrue(rejected_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["total_accepted"], len(accepted_examples))
            self.assertEqual(summary["total_generated"], summary["total_accepted"] + summary["total_rejected"])
            self.assertEqual(summary["rejected_output_path"], str(rejected_path))
            self.assertEqual(summary["validation_report_path"], str(report_path))


if __name__ == "__main__":
    unittest.main()

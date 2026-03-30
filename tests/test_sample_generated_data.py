import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "sample_generated_data.py"


def load_sampling_module():
    spec = importlib.util.spec_from_file_location("sample_generated_data_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SampleGeneratedDataTests(unittest.TestCase):
    def test_sampler_stratifies_by_category_and_label(self) -> None:
        module = load_sampling_module()
        examples = [
            {
                "premise": f"Neg premise entailment {index}.",
                "hypothesis": f"Neg hypothesis entailment {index}.",
                "label": "entailment",
                "category": "negation",
                "generation_method": "method_a",
            }
            for index in range(4)
        ]
        examples.extend(
            {
                "premise": f"Neg premise contradiction {index}.",
                "hypothesis": f"Neg hypothesis contradiction {index}.",
                "label": "contradiction",
                "category": "negation",
                "generation_method": "method_a",
            }
            for index in range(4)
        )
        examples.extend(
            {
                "premise": f"Neg premise neutral {index}.",
                "hypothesis": f"Neg hypothesis neutral {index}.",
                "label": "neutral",
                "category": "negation",
                "generation_method": "method_a",
            }
            for index in range(4)
        )
        examples.extend(
            {
                "premise": f"Temporal premise entailment {index}.",
                "hypothesis": f"Temporal hypothesis entailment {index}.",
                "label": "entailment",
                "category": "temporal/date reasoning",
                "generation_method": "method_b",
            }
            for index in range(3)
        )
        examples.extend(
            {
                "premise": f"Temporal premise contradiction {index}.",
                "hypothesis": f"Temporal hypothesis contradiction {index}.",
                "label": "contradiction",
                "category": "temporal/date reasoning",
                "generation_method": "method_b",
            }
            for index in range(3)
        )

        sampled = module.sample_examples(examples, per_category=6, seed=5)

        self.assertEqual(len(sampled["negation"]), 6)
        negation_labels = [example["label"] for example in sampled["negation"]]
        self.assertGreaterEqual(negation_labels.count("entailment"), 2)
        self.assertGreaterEqual(negation_labels.count("contradiction"), 2)
        self.assertGreaterEqual(negation_labels.count("neutral"), 2)

        self.assertEqual(len(sampled["temporal/date reasoning"]), 6)
        temporal_labels = {example["label"] for example in sampled["temporal/date reasoning"]}
        self.assertEqual(temporal_labels, {"entailment", "contradiction"})

    def test_review_markdown_contains_required_fields(self) -> None:
        module = load_sampling_module()
        sampled = {
            "numeric contradiction": [
                {
                    "premise": "There are five sealed crates on the loading dock.",
                    "hypothesis": "There are five crates on the loading dock.",
                    "label": "entailment",
                    "category": "numeric contradiction",
                    "generation_method": "diverse_targeted_nli_v2",
                }
            ]
        }

        markdown = module.render_review_markdown(
            sampled,
            ROOT / "data" / "generated" / "targeted_nli_v1.json",
            seed=17,
            per_category=12,
        )

        self.assertIn("# Synthetic NLI Review Sample", markdown)
        self.assertIn("## numeric contradiction", markdown)
        self.assertIn("- Category: numeric contradiction", markdown)
        self.assertIn("- Label: entailment", markdown)
        self.assertIn("- Generation method: diverse_targeted_nli_v2", markdown)
        self.assertIn("- Premise: There are five sealed crates on the loading dock.", markdown)
        self.assertIn("- Hypothesis: There are five crates on the loading dock.", markdown)
    def test_sampler_rejects_unknown_label(self) -> None:
        module = load_sampling_module()
        examples = [
            {
                "premise": "A report was filed.",
                "hypothesis": "A report exists.",
                "label": "maybe",
                "category": "negation",
            }
        ]

        with self.assertRaises(ValueError):
            module.sample_examples(examples, per_category=10, seed=7)

    def test_main_writes_review_file_for_wrapped_dataset(self) -> None:
        module = load_sampling_module()
        payload = {
            "examples": [
                {
                    "premise": "The morning briefing started before the audit.",
                    "hypothesis": "The briefing happened first.",
                    "label": "entailment",
                    "category": "temporal/date reasoning",
                    "generation_method": "diverse_targeted_nli_v2",
                },
                {
                    "premise": "No visitor entered the archive after noon.",
                    "hypothesis": "A visitor entered the archive after noon.",
                    "label": "contradiction",
                    "category": "negation",
                    "generation_method": "diverse_targeted_nli_v2",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            input_path = temp_root / "generated.json"
            output_path = temp_root / "review.md"
            input_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            module.main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--per-category",
                    "10",
                    "--seed",
                    "3",
                ]
            )

            review_text = output_path.read_text(encoding="utf-8")
            self.assertIn("## negation", review_text)
            self.assertIn("## temporal/date reasoning", review_text)
            self.assertIn("- Generation method: diverse_targeted_nli_v2", review_text)


if __name__ == "__main__":
    unittest.main()

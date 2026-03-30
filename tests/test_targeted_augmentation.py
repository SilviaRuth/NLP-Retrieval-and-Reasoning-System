import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from data import load_nli_dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "generate_targeted_nli_data.py"


EXPECTED_TEMPLATE_IDS = {
    "negation": {"negation_direct", "negation_scope", "negation_not_all", "negation_none"},
    "numeric contradiction": {"numeric_exact_count", "numeric_comparison", "numeric_bounds", "numeric_parts_total"},
    "temporal/date reasoning": {
        "temporal_before_after",
        "temporal_same_day",
        "temporal_duration",
        "temporal_sequence",
        "temporal_day_relation",
    },
    "long-premise short-hypothesis reasoning": {
        "long_reasoning_transfer",
        "long_reasoning_access",
        "long_reasoning_assignment",
        "long_reasoning_incident",
    },
}
BANNED_TERMS = {"maybe", "perhaps", "probably", "roughly", "approximately"}


def load_generator_module():
    spec = importlib.util.spec_from_file_location("targeted_nli_generator_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TargetedAugmentationTests(unittest.TestCase):
    def test_generation_writes_examples_with_required_metadata_and_summary(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "targeted_nli.json"
            summary_path = Path(temp_dir) / "targeted_nli_summary.json"
            args = module.parse_args(
                [
                    "--negation",
                    "2",
                    "--numeric",
                    "2",
                    "--temporal",
                    "2",
                    "--long_reasoning",
                    "2",
                    "--seed",
                    "7",
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                ]
            )
            examples, summary = module.generate_targeted_dataset(args)

            self.assertEqual(len(examples), 8)
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertEqual(summary["total_generated"], 8)
            self.assertEqual(summary["seed"], 7)
            self.assertEqual(
                summary["counts_by_category"],
                {
                    "negation": 2,
                    "numeric contradiction": 2,
                    "temporal/date reasoning": 2,
                    "long-premise short-hypothesis reasoning": 2,
                },
            )

            saved_examples = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_examples, examples)
            self.assertEqual(len(load_nli_dataset(output_path)), 8)

            required_fields = {
                "premise",
                "hypothesis",
                "label",
                "source",
                "generation_method",
                "category",
                "template_id",
                "seed",
                "validation_status",
            }
            for example in saved_examples:
                self.assertTrue(required_fields.issubset(example.keys()))
                self.assertEqual(example["source"], "synthetic")
                self.assertEqual(example["generation_method"], "diverse_targeted_nli_v2")
                self.assertEqual(example["validation_status"], "passed")

    def test_parse_args_uses_generation_config_defaults(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "generation_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "negation": 4,
                        "numeric": 3,
                        "temporal": 2,
                        "long_reasoning": 1,
                        "seed": 19,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            args = module.parse_args(["--config", str(config_path)])

            self.assertEqual(args.negation, 4)
            self.assertEqual(args.numeric, 3)
            self.assertEqual(args.temporal, 2)
            self.assertEqual(args.long_reasoning, 1)
            self.assertEqual(args.seed, 19)
    def test_generation_is_reproducible_with_config(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            config_path = temp_root / "generation_config.json"
            output_one = temp_root / "run_one.json"
            output_two = temp_root / "run_two.json"
            summary_one = temp_root / "run_one_summary.json"
            summary_two = temp_root / "run_two_summary.json"

            config_path.write_text(
                json.dumps(
                    {
                        "negation": 4,
                        "numeric": 4,
                        "temporal": 5,
                        "long_reasoning": 4,
                        "seed": 19,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            args_one = module.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_one),
                    "--summary-output",
                    str(summary_one),
                ]
            )
            args_two = module.parse_args(
                [
                    "--config",
                    str(config_path),
                    "--output",
                    str(output_two),
                    "--summary-output",
                    str(summary_two),
                ]
            )

            examples_one, summary_payload_one = module.generate_targeted_dataset(args_one)
            examples_two, summary_payload_two = module.generate_targeted_dataset(args_two)

            self.assertEqual(examples_one, examples_two)
            self.assertEqual(summary_payload_one["counts_by_category"], summary_payload_two["counts_by_category"])
            self.assertEqual(summary_payload_one["seed"], 19)
            self.assertEqual(summary_payload_two["seed"], 19)

    def test_generation_covers_expected_subtypes_and_avoids_ambiguous_language(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "diverse_examples.json"
            args = module.parse_args(
                [
                    "--negation",
                    "8",
                    "--numeric",
                    "8",
                    "--temporal",
                    "10",
                    "--long_reasoning",
                    "8",
                    "--seed",
                    "11",
                    "--output",
                    str(output_path),
                ]
            )
            examples, _ = module.generate_targeted_dataset(args)

            seen_templates_by_category: dict[str, set[str]] = {}
            for example in examples:
                seen_templates_by_category.setdefault(example["category"], set()).add(example["template_id"])
                combined = f"{example['premise']} {example['hypothesis']}".lower()
                for term in BANNED_TERMS:
                    self.assertNotIn(term, combined)
                self.assertNotIn("not impossible", combined)
                self.assertNotIn("not uncommon", combined)

            for category_name, expected_templates in EXPECTED_TEMPLATE_IDS.items():
                self.assertTrue(expected_templates.issubset(seen_templates_by_category.get(category_name, set())))

    def test_long_reasoning_examples_stay_multi_sentence_with_short_hypotheses(self) -> None:
        module = load_generator_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "long_reasoning.json"
            args = module.parse_args(
                [
                    "--long_reasoning",
                    "8",
                    "--seed",
                    "23",
                    "--output",
                    str(output_path),
                ]
            )
            examples, _ = module.generate_targeted_dataset(args)

            self.assertEqual(len(examples), 8)
            for example in examples:
                self.assertEqual(example["category"], "long-premise short-hypothesis reasoning")
                self.assertGreaterEqual(example["premise"].count(". "), 2)
                self.assertLessEqual(len(example["hypothesis"].split()), 12)

    def test_generation_refuses_to_write_to_gold_dataset_paths(self) -> None:
        module = load_generator_module()
        protected_output = ROOT / "data" / "train.json"
        args = module.parse_args(["--negation", "1", "--output", str(protected_output)])

        with self.assertRaises(ValueError):
            module.generate_targeted_dataset(args)


if __name__ == "__main__":
    unittest.main()

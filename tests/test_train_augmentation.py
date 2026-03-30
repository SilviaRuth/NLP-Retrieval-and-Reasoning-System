import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from data import NLIExample


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "train.py"


def load_train_module():
    spec = importlib.util.spec_from_file_location("train_under_test", TRAIN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {TRAIN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingAugmentationTests(unittest.TestCase):
    def test_resolve_training_defaults_does_not_auto_enable_augmentation(self) -> None:
        module = load_train_module()
        args = Namespace(
            model_type="bert",
            learning_rate=None,
            gradient_accumulation_steps=None,
            warmup_steps=None,
            warmup_ratio=0.1,
            early_stopping_patience=2,
            augmentation_enabled=False,
            augmentation_path="data/generated/targeted_nli_v1.json",
            augmentation_max_ratio=0.25,
        )

        resolved = module.resolve_training_defaults(args)

        self.assertFalse(resolved.augmentation_enabled)
        self.assertEqual(resolved.learning_rate, 2e-5)
        self.assertEqual(resolved.gradient_accumulation_steps, 4)

    def test_prepare_training_examples_ignores_augmentation_when_disabled(self) -> None:
        module = load_train_module()
        original_examples = [
            NLIExample("A guard checked the south gate.", "A guard inspected a gate.", "entailment"),
            NLIExample("The office stayed closed all day.", "The office opened at noon.", "contradiction"),
        ]
        args = Namespace(
            augmentation_enabled=False,
            augmentation_path="data/generated/targeted_nli_v1.json",
            augmentation_max_ratio=0.25,
            seed=19,
        )

        train_examples, summary = module.prepare_training_examples(args, original_examples)

        self.assertEqual(train_examples, original_examples)
        self.assertEqual(summary["original_training_count"], 2)
        self.assertEqual(summary["synthetic_training_count"], 0)
        self.assertEqual(summary["synthetic_original_ratio"], 0.0)
        self.assertEqual(summary["counts_by_synthetic_category"], {})

    def test_load_augmentation_examples_enforces_ratio_cap_and_counts_categories(self) -> None:
        module = load_train_module()
        payload = {
            "examples": [
                {
                    "premise": "No visitor entered the west archive after lunch.",
                    "hypothesis": "A visitor entered the west archive after lunch.",
                    "label": "contradiction",
                    "category": "negation",
                },
                {
                    "premise": "Not all crates were moved to the front cart.",
                    "hypothesis": "Every crate was moved to the front cart.",
                    "label": "contradiction",
                    "category": "negation",
                },
                {
                    "premise": "Seven forms were signed before noon.",
                    "hypothesis": "Exactly seven forms were signed before noon.",
                    "label": "entailment",
                    "category": "numeric contradiction",
                },
                {
                    "premise": "At most three boxes were left on the platform.",
                    "hypothesis": "Four boxes were left on the platform.",
                    "label": "contradiction",
                    "category": "numeric contradiction",
                },
                {
                    "premise": "The safety drill ended before the equipment review began.",
                    "hypothesis": "The review started after the drill ended.",
                    "label": "entailment",
                    "category": "temporal/date reasoning",
                },
                {
                    "premise": "Mina checked the storage cabinet. She found the seal intact. She reported the result to Omar.",
                    "hypothesis": "Mina reported an intact seal.",
                    "label": "entailment",
                    "category": "long-premise short-hypothesis reasoning",
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "synthetic.json"
            temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            selected_examples, summary = module.load_augmentation_examples(
                str(temp_path),
                original_training_count=4,
                augmentation_max_ratio=0.5,
                seed=7,
            )

        self.assertEqual(len(selected_examples), 2)
        self.assertEqual(summary["original_training_count"], 4)
        self.assertEqual(summary["loaded_synthetic_count"], 6)
        self.assertEqual(summary["synthetic_training_count"], 2)
        self.assertEqual(summary["synthetic_original_ratio"], 0.5)
        self.assertEqual(summary["max_allowed_synthetic_count"], 2)
        self.assertTrue(summary["capped_by_ratio"])
        self.assertEqual(
            summary["loaded_counts_by_synthetic_category"],
            {
                "long-premise short-hypothesis reasoning": 1,
                "negation": 2,
                "numeric contradiction": 2,
                "temporal/date reasoning": 1,
            },
        )
        self.assertEqual(sum(summary["counts_by_synthetic_category"].values()), 2)


if __name__ == "__main__":
    unittest.main()

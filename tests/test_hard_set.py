import json
import tempfile
import unittest
from pathlib import Path

from data import load_nli_dataset
from evaluation.hard_set import extract_error_tags


class HardSetWorkflowTests(unittest.TestCase):
    def test_loads_examples_payload_format(self) -> None:
        payload = {
            "summary": {"total_examples": 1},
            "examples": [
                {
                    "premise": "The device is connected.",
                    "hypothesis": "The device is not connected.",
                    "label": "contradiction",
                }
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            temp_path = Path(handle.name)

        try:
            examples = load_nli_dataset(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].label, "contradiction")

    def test_extracts_negation_long_and_numeric_date_tags(self) -> None:
        tags = extract_error_tags(
            premise="The festival started on 2019-04-01 and no delays were reported.",
            hypothesis="The festival ended before 2019-04-01.",
            primary_category="long_sequence",
        )

        self.assertIn("long_sequence", tags)
        self.assertIn("negation", tags)
        self.assertIn("numeric_date", tags)


if __name__ == "__main__":
    unittest.main()

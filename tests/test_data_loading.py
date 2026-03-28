import json
import tempfile
import unittest
from pathlib import Path

from data import infer_label_map, load_nli_dataset


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA = ROOT / "data" / "sample_nli.json"


class DataLoadingTests(unittest.TestCase):
    def test_loads_list_format_and_normalizes_labels(self) -> None:
        examples = load_nli_dataset(SAMPLE_DATA)
        self.assertEqual(len(examples), 6)
        self.assertEqual(examples[1].label, "entailment")
        self.assertEqual(examples[2].label, "contradiction")

    def test_loads_columnar_format(self) -> None:
        payload = {
            "premise": {"0": "A cat sits on a mat."},
            "hypothesis": {"0": "An animal is resting."},
            "label": {"0": "entails"},
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
            json.dump(payload, handle)
            temp_path = Path(handle.name)

        try:
            examples = load_nli_dataset(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].label, "entailment")

    def test_infers_preferred_label_order(self) -> None:
        examples = load_nli_dataset(SAMPLE_DATA)
        label_to_id = infer_label_map(examples)
        self.assertEqual(list(label_to_id.keys()), ["entailment", "contradiction", "neutral"])


if __name__ == "__main__":
    unittest.main()

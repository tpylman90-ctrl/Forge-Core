"""Dataset-quality gates for independently identifiable train/validation documents."""
import json
import tempfile
import unittest
from pathlib import Path

from forge_core.neural.prepare import Document, clean_documents, prepare


class TestLanguageDataQuality(unittest.TestCase):
    def test_document_manifest_split_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)
            report = prepare("demo", p, 24, 10)
            self.assertEqual(report["schema"], 2)
            self.assertEqual(len(report["train_documents"]), 24)
            self.assertEqual(len(report["valid_documents"]), 10)
            self.assertTrue(all(x["id"].startswith("train:") for x in report["train_documents"]))
            self.assertTrue(all(x["id"].startswith("validation:") for x in report["valid_documents"]))
            self.assertFalse({x["sha256"] for x in report["train_documents"]} &
                             {x["sha256"] for x in report["valid_documents"]})
            self.assertEqual(json.loads((p / "manifest.json").read_text())["train_sha256"],
                             report["train_sha256"])

    def test_duplicate_documents_rejected(self):
        example = "A young gardener found a bright flower by the river and brought the flower to the village school."
        docs, dropped = clean_documents([
            Document("a", example), Document("b", example.upper()),
        ], set(), [])
        self.assertEqual(len(docs), 1)
        self.assertEqual(dropped["exact_duplicate"], 1)

    def test_very_close_paraphrase_rejected(self):
        example = " ".join(["a curious explorer asked many questions about the river and the mountain"] * 8)
        docs, dropped = clean_documents([
            Document("a", example), Document("b", example + " Today."),
        ], set(), [])
        self.assertEqual(len(docs), 1)
        self.assertEqual(dropped["near_duplicate"], 1)

    def test_reject_insufficient_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "independent"):
                prepare("demo", Path(temp), 1, 1)


if __name__ == "__main__":
    unittest.main()

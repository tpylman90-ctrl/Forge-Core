"""Frozen upstream source slice parser tests; no live network in CI."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from forge_core.neural.pinned_corpus import clean_prefix, prepare, source_url, REVISION


class TestPinnedCorpus(unittest.TestCase):
    def test_pin_explicit_and_no_path_injection(self):
        self.assertIn(REVISION, source_url("TinyStories-train.txt"))
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            source_url("../../private")

    def test_distinct_public_documents_and_exact_hashes(self):
        text = "".join(f"Once upon a time there was a little fox with friend {i}." +
                       "<|endoftext|>" for i in range(36))
        valid = "".join(f"A cheerful bird played outside with neighbors number {i}." +
                        "<|endoftext|>" for i in range(38))
        def fake_fetch(name, limit):
            return (text if name.endswith("-train.txt") else valid).encode()
        with patch("forge_core.neural.pinned_corpus.fetch_prefix", side_effect=fake_fetch):
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                m = prepare(out, 200000, 200000)
                self.assertGreater(m["train_document_count"], 30)
                self.assertGreater(m["valid_document_count"], 30)
                self.assertNotEqual(m["train_sha256"], m["valid_sha256"])
                self.assertEqual(m["train_sha256"],
                    hashlib.sha256((out/"train.txt").read_bytes()).hexdigest())

    def test_cross_split_story_overlap_is_filtered(self):
        train = "".join(f"Once upon a time the small fox met friend {i} in the forest." +
                        "<|endoftext|>" for i in range(40))
        valid = (train.split("<|endoftext|>")[0] + "<|endoftext|>" +
                 "".join(f"The little bird visited a new garden number {i}." +
                         "<|endoftext|>" for i in range(40)))
        with patch("forge_core.neural.pinned_corpus.fetch_prefix",
                   side_effect=lambda name, limit:
                   (train if name.endswith("-train.txt") else valid).encode()):
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder)
                report = prepare(out, 200000, 200000)
                self.assertEqual(report["validation_records_removed_as_duplicate_or_train_overlap"], 1)
                self.assertNotIn("fox met friend", (out / "valid.txt").read_text())

    def test_incomplete_records_removed(self):
        text = "".join(f"A little story {i} with enough characters to pass the rules." +
                       "<|endoftext|>" for i in range(34))
        data = (text + "THIS IS A CUT-OFF FRAGMENT").encode()
        cleaned, n = clean_prefix(data)
        self.assertEqual(n, 34)
        self.assertNotIn("CUT-OFF", cleaned)


if __name__ == "__main__":
    unittest.main()

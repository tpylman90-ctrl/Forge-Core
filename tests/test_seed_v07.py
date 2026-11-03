"""BPE from scratch: corpus pin, token roundtrip, checkpoint and tiny training."""
import tempfile
import unittest
from pathlib import Path
from forge_core.neural.seed_v07 import learn_tokenizer, train
import json
import hashlib


class TestSeedV07(unittest.TestCase):
    def test_from_scratch_tokenizer_roundtrip_and_tiny_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data"
            dataset.mkdir()
            train_text = ("Once upon a time a fox found a kind rabbit. " * 125 +
                          "\n\nThen the little dog walked home with his friend.\n\n") * 2
            valid_text = ("A very happy cat went to the green garden. " * 135 +
                          "\n\nThe friendly teacher told a short story.\n\n") * 2
            (dataset/"train.txt").write_text(train_text)
            (dataset/"valid.txt").write_text(valid_text)
            manifest = {"schema":1,
                "revision":"f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
                "license":"cdla-sharing-1.0","train_document_count":200,
                "valid_document_count":200,
                "train_sha256":hashlib.sha256(train_text.encode()).hexdigest(),
                "valid_sha256":hashlib.sha256(valid_text.encode()).hexdigest()}
            (dataset/"manifest.json").write_text(json.dumps(manifest))
            codec=learn_tokenizer(dataset/"train.txt",root/"codec.json",vocab=512)
            text="Once upon a time, the fox came home."
            self.assertEqual(codec.decode(codec.encode(text).ids),text)
            out=root/"out"
            result=train(source=dataset,out=out,steps=2,model_size="tiny",
                         batch=2,seq_len=32,vocab=512,eval_every=1)
            self.assertEqual(result["steps_total"],2)
            self.assertTrue((out/"checkpoint.pt").exists())
            resumed=train(source=dataset,out=out,steps=1,model_size="tiny",
                          batch=2,seq_len=32,vocab=512,eval_every=1,
                          resume=out/"checkpoint.pt")
            self.assertEqual(resumed["steps_total"],3)
            (dataset/"valid.txt").write_text("altered")
            with self.assertRaisesRegex(ValueError,"valid data differs"):
                train(source=dataset,out=out,steps=1,model_size="tiny",
                      batch=2,seq_len=32,vocab=512,eval_every=1)


if __name__ == "__main__":
    unittest.main()

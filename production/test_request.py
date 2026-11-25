import json
import tempfile
import unittest
from pathlib import Path
from request import load_request

class RequestTests(unittest.TestCase):
    def setUp(self):
        self.spec=json.loads((Path(__file__).parent/"requests/cargo_crate.json").read_text())
    def read(self,value):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"input.json"
            p.write_text(json.dumps(value))
            return load_request(p)
    def test_valid(self):
        self.assertEqual(self.read(self.spec),self.spec)
    def test_unknown_fields(self):
        with self.assertRaises(ValueError):
            self.read(dict(self.spec, python="arbitrary code"))
    def test_traversal(self):
        with self.assertRaises(ValueError):
            self.read(dict(self.spec,asset_id="../secret"))
    def test_family(self):
        with self.assertRaises(ValueError):
            self.read(dict(self.spec,family="arbitrary"))
    def test_nonfinite(self):
        with self.assertRaises(ValueError):
            self.read(dict(self.spec,dimensions_m=[1,float("nan"),1]))
    def test_limits(self):
        for changed in (dict(texture_size=4096),dict(triangle_budget=1),dict(seed=True),dict(dimensions_m=[0,1,1])):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.read(dict(self.spec,**changed))

if __name__=="__main__":
    unittest.main()

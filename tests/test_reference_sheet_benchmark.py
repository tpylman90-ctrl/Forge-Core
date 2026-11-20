import importlib.util
from pathlib import Path
import json
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_reference_sheet_benchmark.py"
SPEC = importlib.util.spec_from_file_location("reference_sheet_benchmark", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)


def valid_spec():
    return {
        "schema_version": 1,
        "asset_id": "synthetic-fixture",
        "units": "m",
        "coordinate_frame": {"x": "left-right", "y": "front-rear", "z": "up", "forward_axis": "+Y"},
        "dimensions": {"length": 8.0, "width": 4.0, "height": 2.0},
        "views": {
            "front": {"crop_norm": [0.0, 0.0, 0.2, 0.2], "confidence": 0.9},
            "top": {"crop_norm": [0.2, 0.0, 0.2, 0.2], "confidence": 0.9},
            "side": {"crop_norm": [0.4, 0.0, 0.2, 0.2], "confidence": 0.9},
            "three_quarter": {"crop_norm": [0.6, 0.0, 0.2, 0.2], "confidence": 0.8},
        },
        "curvature": {"present": True, "legend_present": False, "color_semantics": None},
    }


class ReferenceSheetValidationTests(unittest.TestCase):
    def test_accepts_valid_sheet_spec_without_curvature_legend(self):
        module.validate_spec(valid_spec())

    def test_rejects_curvature_interpretation_without_legend(self):
        spec = valid_spec()
        spec["curvature"]["color_semantics"] = {"red": "convex"}
        with self.assertRaisesRegex(ValueError, "cannot be interpreted"):
            module.validate_spec(spec)

    def test_rejects_crop_that_escapes_image(self):
        spec = valid_spec()
        spec["views"]["front"]["crop_norm"] = [0.9, 0.0, 0.2, 0.2]
        with self.assertRaisesRegex(ValueError, "stay inside"):
            module.validate_spec(spec)

    def test_rejects_missing_required_view(self):
        spec = valid_spec()
        del spec["views"]["side"]
        with self.assertRaisesRegex(ValueError, "missing required views"):
            module.validate_spec(spec)

    def test_aspect_error_is_scale_independent(self):
        error = module.sorted_aspect_error([16.0, 8.0, 4.0], {"length": 8.0, "width": 4.0, "height": 2.0})
        self.assertEqual(error, 0.0)

    def test_crops_synthetic_linked_views_without_touching_source(self):
        from PIL import Image

        spec = valid_spec()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "sheet.png"
            Image.new("RGB", (100, 100), "white").save(source)
            spec["source_sheet"] = source.name
            spec_path = root / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            crops = module.crop_views(spec, spec_path, root / "crops")
            self.assertEqual(set(crops), {"front", "top", "side", "three_quarter"})
            self.assertTrue(all(path.is_file() for path in crops.values()))
            self.assertTrue(source.is_file())

    def test_rejects_source_sheet_path_escape(self):
        spec = valid_spec()
        spec["source_sheet"] = "../outside.png"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "must stay inside"):
                module.crop_views(spec, root / "spec.json", root / "crops")


if __name__ == "__main__":
    unittest.main()

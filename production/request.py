"""Strict, dependency-free asset request parser. No executable input."""
import json
import math
import re
from pathlib import Path

def load_request(path):
    path = Path(path)
    if path.stat().st_size > 16384:
        raise ValueError("Request exceeds 16 KiB")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key: " + key)
            result[key] = value
        return result
    spec = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    expected = {"schema_version", "asset_id", "family", "dimensions_m", "color", "seed", "triangle_budget", "texture_size"}
    if not isinstance(spec, dict) or set(spec) != expected:
        raise ValueError("Unexpected or missing request keys")
    if type(spec["schema_version"]) is not int or spec["schema_version"] != 1 or spec["family"] != "cargo_crate":
        raise ValueError("Unsupported request version or family")
    if not isinstance(spec["asset_id"], str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", spec["asset_id"]):
        raise ValueError("Invalid asset identifier")
    for key, low, high in (("dimensions_m", 0.35, 5.0), ("color", 0.0, 1.0)):
        value = spec[key]
        if not isinstance(value, list) or len(value) != 3 or any(type(x) not in (float, int) or not math.isfinite(x) or not low <= x <= high for x in value):
            raise ValueError("Invalid " + key)
    for key, low, high in (("seed", 0, 2147483647), ("triangle_budget", 1000, 50000)):
        if type(spec[key]) is not int or not low <= spec[key] <= high:
            raise ValueError("Invalid " + key)
    if type(spec["texture_size"]) is not int or spec["texture_size"] not in (128, 256, 512, 1024):
        raise ValueError("Unsupported texture size")
    return spec

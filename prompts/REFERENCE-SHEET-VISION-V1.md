# Reference-sheet vision prompt v1

Use this prompt with the image-capable GPT system at runtime. It produces a
structured interpretation for the 3D tools; it does not train or update model
weights. Keep the supplied image and returned JSON in the user's private
project/Kaggle inputs.

```text
Read the attached technical 3D reference sheet as linked views of one object.
Return JSON only, following the schema below. Do not invent geometry hidden
from every view. Use null plus a short uncertainty note when evidence is
missing or ambiguous.

Identify the labeled views and give each view's crop as normalized coordinates
[left, top, width, height] from 0.0 to 1.0 relative to the full image. Include
the labels, object orientation, and confidence. Link corresponding landmarks
across views (nose, rear, centerline, cockpit, pods, wings/fins, intakes,
thrusters, vents, and major panel breaks). Treat dimensions printed on the
sheet as measurements; do not infer scale from pixels when a measurement is
present. State units and the axis convention. Check whether the dimensions
agree across views and list conflicts.

For false-color curvature or normal maps, report whether a legend is visible.
If there is no legend, set color_semantics to null and do not infer convex,
concave, or curvature magnitude from the colors. Separate real surfaces and
part boundaries from wireframe edges, annotations, and measurement lines.

The output must match this shape:
{
  "schema_version": 1,
  "asset_id": "short descriptive id",
  "source_sheet": "reference.jpg",
  "units": "m",
  "coordinate_frame": {"x": "...", "y": "...", "z": "...", "forward_axis": "..."},
  "dimensions": {"length": 0.0, "width": 0.0, "height": 0.0},
  "views": {
    "front": {"crop_norm": [0,0,0,0], "label": "...", "confidence": 0.0},
    "rear": {"crop_norm": [0,0,0,0], "label": "...", "confidence": 0.0},
    "top": {"crop_norm": [0,0,0,0], "label": "...", "confidence": 0.0},
    "side": {"crop_norm": [0,0,0,0], "label": "...", "confidence": 0.0},
    "three_quarter": {"crop_norm": [0,0,0,0], "label": "...", "confidence": 0.0}
  },
  "parts": [{"id": "...", "observed_in": ["side"], "landmarks": ["..."], "confidence": 0.0}],
  "curvature": {"present": false, "legend_present": false, "color_semantics": null},
  "uncertainties": ["..."]
}

Use null for an unavailable measurement. A required view that is absent should
be omitted and named in uncertainties. Do not fabricate a crop for it.
```

The Kaggle benchmark requires at least a front, top, side, and one three-quarter
view to run the multi-view comparison. Rear view is strongly preferred. The
curvature guard is intentional: a color image without a legend must remain
uninterpreted rather than receive a guessed convex/concave label.

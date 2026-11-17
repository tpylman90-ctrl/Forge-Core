# Forge semantic parts and control-family holdouts v4

Train from random initialization using the CC0 multi-view RGB and semantic part-map dataset. Include all six categories in training and validation. Train on normalized geometry and detail controls p0–p6 in the central 0.20–0.80 range; test on both central samples and parameter-extreme samples outside that range, split across proportion controls p0–p2 and detail controls p3–p6. Keep p7 fixed and excluded.

Run seeds 43, 53, and 73. Report category accuracy, active geometry MAE, foreground part mIoU, and Blender silhouette IoU for seen and held-out control families, across text-only, single-view, four-view, and combined modes. Preserve all lineage, fixed-split hashes, candidate checkpoints, and metrics. Keep this branch separate from Forge Seed weights and research-only.

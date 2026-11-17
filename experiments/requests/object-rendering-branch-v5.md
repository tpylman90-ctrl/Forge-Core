# Forge compositional shape and detail holdout v5

Train all six categories and semantic part maps. In each training object, use central controls or at most one extreme p0-p6 control; do not show multi-axis extreme combinations. The held-out test family combines two or three extreme proportion/detail controls on objects across all categories. Run seeds 43, 53, and 73.

Report category accuracy, active geometry MAE, foreground part mIoU, and seen versus multi-axis-held-out Blender silhouette IoU across text, single-view, four-view, and combined modes. Use the fixed CC0 split and preserve dataset and checkpoint lineage. Keep the visual model fresh-init and research-only, separate from Forge Seed weights.

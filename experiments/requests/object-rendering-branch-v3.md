# Forge semantic parts and held-out families v3

Generate CC0 multi-view images with semantic pixel maps for body, support, feature, trim, and background. Train fresh isolated seeds 43, 53, and 73 on crate, stool, lamp, and hovercraft; withhold chair and mug from both training and validation. Add a segmentation head and train against the part maps while retaining active geometry parameter regression.

Report seen-family and held-out-family category, geometry MAE, and foreground part mIoU for text-only, single-view, four-view, and combined inputs. Render predictions against held-out Blender targets and report silhouette IoU separately for seen families and withheld chair/mug families. Keep all weights research-only, separate from the Forge Seed language model, and do not promote or integrate until held-out part and shape metrics are repeatable.

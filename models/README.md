# Model Artifacts Directory

This directory stores serialized models, preprocessing pipelines, and audit metadata.

## Artifact Catalog

- `document_verifier.joblib`: Serialized classical classifier (`HistGradientBoosting` with L2 regularization and calibrated threshold 0.25).
- `feature_pipeline.joblib`: Fitted `StandardScaler` (fit strictly on the training partition) and ordered 37-feature name list.
- `metadata.json`: Complete audit record of the classical model, including validation threshold (0.25), validation F1 (0.3622), test metrics (F1 0.2065, Recall 45.71%), and baseline Logistic Regression metrics for comparison.
- `visual_verifier_resnet18.pt`: Pretrained `ResNet-18` transfer learning checkpoint fine-tuned on document receipts with class-weighted cross-entropy loss and Grad-CAM hooks.
- `visual_metadata.json`: Architecture specification, optimal validation threshold (0.48), validation metrics (F1 0.3067, Recall 69.70%), and test metrics (F1 0.2436, Recall 54.29%).

## Regeneration Commands

To retrain and regenerate these checkpoints:
```bash
# Classical tabular pipeline
python train.py

# Visual transfer learning model
python train_visual.py
```

"""Phase 6: Final Unbiased Test Set Evaluation.

Evaluates the candidate selected via Validation Forged-Class F1 strictly ONCE
on the untouched Test Set (218 samples).

Also evaluates the visual transfer learning model ONCE on the test set
for comprehensive scientific comparison.

CRITICAL RULE:
The test set was kept completely untouched during feature diagnosis, model training,
hyperparameter tuning, and validation threshold selection.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DATASET_ROOT
from src.data_loader import DatasetLoader
from src.visual_model import DocumentDataset, DocumentForgeryResNet, get_visual_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_selected")


def evaluate_tabular_test():
    """Evaluate selected classical model on untouched test set."""
    train_data = np.load(CACHE_DIR / "features_train.npz")
    test_data = np.load(CACHE_DIR / "features_test.npz")

    x_train, y_train = train_data["x"], train_data["y"]
    x_test, y_test = test_data["x"], test_data["y"]

    # Fit selected model on train data
    clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        l2_regularization=10.0,
        min_samples_leaf=20,
        max_iter=100,
        random_state=42,
    )
    clf.fit(x_train, y_train)

    y_test_prob = clf.predict_proba(x_test)[:, 1]

    # Evaluate at default threshold (0.50)
    y_test_pred_def = (y_test_prob >= 0.50).astype(int)
    # Evaluate at optimal validation threshold (0.25)
    y_test_pred_opt = (y_test_prob >= 0.25).astype(int)

    def calc(y_true, y_pred, y_prob):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_forged": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "recall_forged": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "f1_forged": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    res_default = calc(y_test, y_test_pred_def, y_test_prob)
    res_opt = calc(y_test, y_test_pred_opt, y_test_prob)

    return clf, y_test, y_test_pred_opt, y_test_prob, res_default, res_opt


def evaluate_visual_test():
    """Evaluate ResNet-18 model on untouched test set."""
    ckpt_path = MODELS_DIR / "visual_verifier_resnet18.pt"
    if not ckpt_path.is_file():
        return None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    image_size = ckpt.get("image_size", 384)
    opt_th = ckpt.get("optimal_threshold", 0.48)

    loader = DatasetLoader(DATASET_ROOT)
    _, _, test_records = loader.load_all_splits()

    _, test_transform = get_visual_transforms(image_size=image_size)
    test_ds = DocumentDataset(test_records, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    model = DocumentForgeryResNet(pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels, _ in test_loader:
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.numpy())
            all_labels.extend(labels.numpy())

    y_test = np.array(all_labels)
    y_test_prob = np.array(all_probs)

    y_pred_def = (y_test_prob >= 0.50).astype(int)
    y_pred_opt = (y_test_prob >= opt_th).astype(int)

    def calc(y_true, y_pred, y_prob):
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision_forged": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "recall_forged": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "f1_forged": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    return opt_th, calc(y_test, y_pred_def, y_test_prob), calc(y_test, y_pred_opt, y_test_prob)


def main():
    print("\n" + "=" * 85)
    print("      PHASE 6: FINAL TEST SET EVALUATION ON UNTOUCHED SPLIT (218 samples)")
    print("=" * 85)

    clf, y_test, y_pred_opt, y_prob, tab_def, tab_opt = evaluate_tabular_test()

    print("\n1. SELECTED MODEL: HistGradientBoosting (L2=10.0, min_leaf=20)")
    print("-" * 85)
    print("Default Threshold (0.50):")
    print(f"  Test Accuracy:        {tab_def['accuracy']:.4f}")
    print(f"  Test Forged Prec:     {tab_def['precision_forged']:.4f}")
    print(f"  Test Forged Recall:   {tab_def['recall_forged']:.4f}")
    print(f"  Test Forged F1:       {tab_def['f1_forged']:.4f}")
    print(f"  Test ROC-AUC:         {tab_def['roc_auc']:.4f}")
    print(f"  Confusion Matrix:     TN={tab_def['confusion_matrix'][0][0]}, FP={tab_def['confusion_matrix'][0][1]}")
    print(f"                        FN={tab_def['confusion_matrix'][1][0]}, TP={tab_def['confusion_matrix'][1][1]}")
    print("-" * 85)
    print("Optimal Validation Threshold (0.25):")
    print(f"  Test Accuracy:        {tab_opt['accuracy']:.4f}")
    print(f"  Test Forged Prec:     {tab_opt['precision_forged']:.4f}")
    print(f"  Test Forged Recall:   {tab_opt['recall_forged']:.4f}")
    print(f"  Test Forged F1:       {tab_opt['f1_forged']:.4f}")
    print(f"  Test ROC-AUC:         {tab_opt['roc_auc']:.4f}")
    print(f"  Confusion Matrix:     TN={tab_opt['confusion_matrix'][0][0]}, FP={tab_opt['confusion_matrix'][0][1]}")
    print(f"                        FN={tab_opt['confusion_matrix'][1][0]}, TP={tab_opt['confusion_matrix'][1][1]}")

    vis_res = evaluate_visual_test()
    if vis_res:
        vis_th, vis_def, vis_opt = vis_res
        print("\n2. VISUAL TRANSFER LEARNING MODEL: ResNet-18")
        print("-" * 85)
        print("Default Threshold (0.50):")
        print(f"  Test Accuracy:        {vis_def['accuracy']:.4f}")
        print(f"  Test Forged Prec:     {vis_def['precision_forged']:.4f}")
        print(f"  Test Forged Recall:   {vis_def['recall_forged']:.4f}")
        print(f"  Test Forged F1:       {vis_def['f1_forged']:.4f}")
        print(f"  Test ROC-AUC:         {vis_def['roc_auc']:.4f}")
        print("-" * 85)
        print(f"Optimal Validation Threshold ({vis_th:.2f}):")
        print(f"  Test Accuracy:        {vis_opt['accuracy']:.4f}")
        print(f"  Test Forged Prec:     {vis_opt['precision_forged']:.4f}")
        print(f"  Test Forged Recall:   {vis_opt['recall_forged']:.4f}")
        print(f"  Test Forged F1:       {vis_opt['f1_forged']:.4f}")
        print(f"  Test ROC-AUC:         {vis_opt['roc_auc']:.4f}")

    # Save final test evaluation JSON
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    final_report = {
        "selected_model": "HistGradientBoosting (l2=10.0, min_leaf=20, threshold=0.25)",
        "selected_test_metrics_default_0_5": tab_def,
        "selected_test_metrics_optimized_0_25": tab_opt,
        "visual_model_test_metrics_default_0_5": vis_def if vis_res else None,
        "visual_model_test_metrics_optimized": vis_opt if vis_res else None,
    }
    with open(OUTPUTS_DIR / "final_test_evaluation.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Saved final test evaluation report to final_test_evaluation.json")


if __name__ == "__main__":
    main()

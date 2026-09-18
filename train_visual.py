"""Phase 4 & Phase 5: Training and Evaluation Pipeline for Visual Transfer Learning Model.

Uses ResNet-18 pretrained backbone:
- Frozen backbone warmup (Stage 1) + upper layer fine-tuning (Stage 2)
- Training-only realistic data augmentation
- Class-weighted loss for ~5:1 imbalance
- Evaluated on Validation set for checkpoint and threshold selection
- Test set remains completely untouched until final evaluation
"""

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from src.config import DATASET_ROOT, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import DatasetLoader
from src.visual_model import DocumentDataset, DocumentForgeryResNet, get_visual_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_visual")


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Run inference over a DataLoader and return loss, predicted probabilities, and true labels."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def calculate_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Compute performance metrics given true labels and probabilities at a decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.0

    return {
        "threshold": round(threshold, 3),
        "accuracy": round(acc, 4),
        "precision_forged": round(prec_1, 4),
        "recall_forged": round(rec_1, 4),
        "f1_forged": round(f1_1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
    }


def find_optimal_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find threshold maximizing validation Forged F1."""
    default_m = calculate_metrics_at_threshold(y_val, y_val_prob, threshold=0.50)
    best_m = default_m
    best_key = (default_m["f1_forged"], default_m["recall_forged"], default_m["roc_auc"])

    for th in np.arange(0.05, 0.95, 0.01):
        m = calculate_metrics_at_threshold(y_val, y_val_prob, threshold=float(th))
        key = (m["f1_forged"], m["recall_forged"], m["roc_auc"])
        if key > best_key:
            best_key = key
            best_m = m

    return default_m, best_m


def main():
    parser = argparse.ArgumentParser(description="Train Visual Transfer Learning Model (ResNet-18)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Epochs for training classification head")
    parser.add_argument("--stage2-epochs", type=int, default=7, help="Epochs for fine-tuning layer4 + head")
    parser.add_argument("--image-size", type=int, default=384, help="Input image dimension (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # 1. Load Dataset Records
    loader = DatasetLoader(DATASET_ROOT)
    train_records, val_records, test_records = loader.load_all_splits()
    logger.info("Splits loaded: Train=%d, Val=%d, Test=%d", len(train_records), len(val_records), len(test_records))

    # 2. Data Transforms & Loaders
    train_transform, val_transform = get_visual_transforms(image_size=args.image_size)

    train_ds = DocumentDataset(train_records, transform=train_transform)
    val_ds = DocumentDataset(val_records, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Model & Loss Function
    model = DocumentForgeryResNet(pretrained=True, dropout=0.3).to(device)

    # Compute positive class weight for ~5:1 imbalance: 483 / 94 ≈ 5.14
    n_gen = sum(1 for r in train_records if r.label == 0)
    n_forg = sum(1 for r in train_records if r.label == 1)
    pos_weight = float(n_gen) / max(float(n_forg), 1.0)
    loss_weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)

    logger.info("Class imbalance weights: Genuine=1.00, Forged=%.2f", pos_weight)

    best_val_f1 = -1.0
    best_checkpoint_state = None
    best_epoch = 0

    print("\n" + "=" * 85)
    print("      STAGE 1: WARMUP CLASSIFIER HEAD (Backbone Frozen)")
    print("=" * 85)
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-3)

    for epoch in range(1, args.stage1_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_ds)
        val_loss, val_probs, val_labels = evaluate_model(model, val_loader, device)
        _, val_metrics = find_optimal_threshold(val_labels, val_probs)

        logger.info(
            "Stage 1 Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Val Forged F1: %.4f (th=%.2f) | Val ROC-AUC: %.4f",
            epoch, args.stage1_epochs, train_loss, val_loss, val_metrics["f1_forged"], val_metrics["threshold"], val_metrics["roc_auc"]
        )

        if val_metrics["f1_forged"] > best_val_f1:
            best_val_f1 = val_metrics["f1_forged"]
            best_checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    print("\n" + "=" * 85)
    print("      STAGE 2: FINE-TUNING UPPER CONVOLUTIONAL LAYERS (layer4 + Head)")
    print("=" * 85)
    model.unfreeze_upper_layers()

    # Differential learning rate: lower for layer4, higher for head
    optimizer = torch.optim.AdamW([
        {"params": model.layer4.parameters(), "lr": 1e-4, "weight_decay": 1e-3},
        {"params": model.classifier.parameters(), "lr": 5e-4, "weight_decay": 1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=1e-6)

    for epoch in range(1, args.stage2_epochs + 1):
        global_epoch = args.stage1_epochs + epoch
        model.train()
        train_loss = 0.0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_ds)
        val_loss, val_probs, val_labels = evaluate_model(model, val_loader, device)
        _, val_metrics = find_optimal_threshold(val_labels, val_probs)

        logger.info(
            "Stage 2 Epoch %d/%d (Global %d) | Train Loss: %.4f | Val Loss: %.4f | Val Forged F1: %.4f (th=%.2f) | Val ROC-AUC: %.4f",
            epoch, args.stage2_epochs, global_epoch, train_loss, val_loss, val_metrics["f1_forged"], val_metrics["threshold"], val_metrics["roc_auc"]
        )

        if val_metrics["f1_forged"] > best_val_f1:
            best_val_f1 = val_metrics["f1_forged"]
            best_checkpoint_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = global_epoch

    # Load best validation checkpoint
    logger.info("Loading best model checkpoint from Epoch %d (Validation Forged F1: %.4f)", best_epoch, best_val_f1)
    model.load_state_dict({k: v.to(device) for k, v in best_checkpoint_state.items()})

    # Detailed validation evaluation on best checkpoint
    _, final_val_probs, final_val_labels = evaluate_model(model, val_loader, device)
    default_val_m, opt_val_m = find_optimal_threshold(final_val_labels, final_val_probs)

    print("\n" + "=" * 85)
    print("           VISUAL MODEL VALIDATION RESULTS (ResNet-18 Transfer Learning)")
    print("=" * 85)
    print(f"Default Threshold (0.50):")
    print(f"  Val Accuracy:       {default_val_m['accuracy']:.4f}")
    print(f"  Val Forged Prec:    {default_val_m['precision_forged']:.4f}")
    print(f"  Val Forged Recall:  {default_val_m['recall_forged']:.4f}")
    print(f"  Val Forged F1:      {default_val_m['f1_forged']:.4f}")
    print(f"  Val ROC-AUC:        {default_val_m['roc_auc']:.4f}")
    print("-" * 85)
    print(f"Optimal Validation Threshold ({opt_val_m['threshold']:.2f}):")
    print(f"  Val Accuracy:       {opt_val_m['accuracy']:.4f}")
    print(f"  Val Forged Prec:    {opt_val_m['precision_forged']:.4f}")
    print(f"  Val Forged Recall:  {opt_val_m['recall_forged']:.4f}")
    print(f"  Val Forged F1:      {opt_val_m['f1_forged']:.4f}  <-- PRIMARY SELECTION SCORE")
    print(f"  Val ROC-AUC:        {opt_val_m['roc_auc']:.4f}")
    print("=" * 85)

    # Save visual model artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    visual_model_path = MODELS_DIR / "visual_verifier_resnet18.pt"
    visual_meta_path = MODELS_DIR / "visual_metadata.json"

    torch.save({
        "model_state_dict": best_checkpoint_state,
        "image_size": args.image_size,
        "best_epoch": best_epoch,
        "optimal_threshold": opt_val_m["threshold"],
        "default_metrics": default_val_m,
        "optimized_metrics": opt_val_m,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, visual_model_path)
    logger.info("Saved visual model weights to %s", visual_model_path)

    report = {
        "model_architecture": "ResNet18 (Transfer Learning)",
        "pretrained_source": "torchvision.models.ResNet18_Weights.DEFAULT",
        "best_epoch": best_epoch,
        "image_size": args.image_size,
        "optimal_validation_threshold": opt_val_m["threshold"],
        "validation_metrics_default_0_5": default_val_m,
        "validation_metrics_optimized": opt_val_m,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(visual_meta_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / "visual_model_val_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

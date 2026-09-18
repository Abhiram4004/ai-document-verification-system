"""Model training and evaluation pipeline for AI Document Verification System."""

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CACHE_DIR, DATASET_ROOT, MODELS_DIR, OUTPUTS_DIR
from src.data_loader import DatasetLoader, DocumentRecord, get_split_statistics
from src.feature_extraction import FEATURE_NAMES, FeatureExtractor
from src.model import (
    ModelTrainer,
    calculate_metrics,
    get_model_interpretation,
    save_pipeline,
)
from src.ocr import OCRPipeline, ensure_tesseract_available

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train")


def extract_features_for_split(
    records: List[DocumentRecord],
    split_name: str,
    extractor: FeatureExtractor,
    cache_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract or load cached feature matrix and labels for a split."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"features_{split_name}.npz"

    # Check if cached feature matrix exists and matches record count
    if cache_file.is_file():
        try:
            cached_data = np.load(cache_file, allow_pickle=True)
            x_cached = cached_data["x"]
            y_cached = cached_data["y"]
            cached_filenames = list(cached_data["filenames"])
            current_filenames = [r.filename for r in records]

            if cached_filenames == current_filenames and x_cached.shape[0] == len(records):
                logger.info(
                    "Loaded cached %s features: %d samples, %d features",
                    split_name.upper(),
                    x_cached.shape[0],
                    x_cached.shape[1],
                )
                return x_cached, y_cached
        except Exception as e:
            logger.warning("Failed reading cached %s features: %s. Recomputing...", split_name, e)

    logger.info("Extracting features for %s split (%d samples)...", split_name.upper(), len(records))
    x_list = []
    y_list = []
    filenames = []

    total = len(records)
    for i, record in enumerate(records, 1):
        if i % 50 == 0 or i == total:
            logger.info("  [%s] Processed %d / %d documents (%.1f%%)", split_name.upper(), i, total, (i / total) * 100)
            extractor.ocr_pipeline.save_cache()

        feat_vector, _, _ = extractor.extract_features(
            record.image_path,
            cache_key=record.filename,
        )

        x_list.append(feat_vector)
        y_list.append(record.label)
        filenames.append(record.filename)

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    # Save to disk cache
    try:
        np.savez_compressed(cache_file, x=x, y=y, filenames=np.array(filenames))
        logger.info("Saved %s feature cache to %s", split_name.upper(), cache_file.name)
    except Exception as e:
        logger.warning("Could not save feature cache: %s", e)

    # Save OCR cache after processing split
    extractor.ocr_pipeline.save_cache()

    return x, y


def save_evaluation_plots(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model: any,
    feature_names: List[str],
    outputs_dir: Path,
) -> None:
    """Generate and save confusion matrix, ROC curve, and feature importances."""
    outputs_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

        # 1. Confusion Matrix Plot
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["Genuine (0)", "Forged (1)"],
            cmap="Blues",
            values_format="d",
            ax=ax,
        )
        ax.set_title("Test Set Confusion Matrix - Document Verification", fontsize=12, fontweight="bold")
        plt.tight_layout()
        cm_path = outputs_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=200)
        plt.close(fig)
        logger.info("Saved confusion matrix plot to %s", cm_path.name)

        # 2. ROC Curve Plot
        if len(np.unique(y_test)) > 1 and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color="#2563eb", lw=2, label="ROC curve")
            ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Random Chance")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate (1 - Specificity)")
            ax.set_ylabel("True Positive Rate (Sensitivity / Recall)")
            ax.set_title("Test Set ROC Curve", fontsize=12, fontweight="bold")
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            roc_path = outputs_dir / "roc_curve.png"
            fig.savefig(roc_path, dpi=200)
            plt.close(fig)
            logger.info("Saved ROC curve plot to %s", roc_path.name)

        # 3. Model Interpretation Plot (if supported)
        interp = get_model_interpretation(model, feature_names)
        if interp["type"] in ("feature_importance", "coefficients") and interp["values"]:
            top_vals = interp["values"][:15]
            feats = [item["feature"] for item in reversed(top_vals)]
            scores = [item.get("importance", item.get("coefficient", 0.0)) for item in reversed(top_vals)]

            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#3b82f6" if s >= 0 else "#ef4444" for s in scores]
            ax.barh(feats, scores, color=colors)
            val_type = "Importance" if interp["type"] == "feature_importance" else "Coefficient"
            ax.set_xlabel(f"Relative {val_type}")
            ax.set_title(f"Top 15 Predictive Features ({interp['title']})", fontsize=12, fontweight="bold")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            imp_path = outputs_dir / "feature_importance.png"
            fig.savefig(imp_path, dpi=200)
            plt.close(fig)
            logger.info("Saved feature interpretation plot to %s", imp_path.name)

    except Exception as e:
        logger.warning("Could not generate evaluation plots: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Train AI Document Verification Classifier")
    parser.add_argument("--dataset-dir", type=str, default=str(DATASET_ROOT), help="Path to Find it Again dataset")
    parser.add_argument("--cache-dir", type=str, default=str(CACHE_DIR), help="Feature cache directory")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Model output directory")
    parser.add_argument("--outputs-dir", type=str, default=str(OUTPUTS_DIR), help="Evaluation output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("\n" + "=" * 75)
    print("       AI DOCUMENT VERIFICATION SYSTEM: TRAINING & EVALUATION")
    print("=" * 75)

    # 1. Mandatory Runtime Check: Tesseract OCR
    logger.info("Checking Tesseract OCR runtime dependency...")
    tesseract_cmd = ensure_tesseract_available()
    logger.info("Verified Tesseract OCR binary: %s", tesseract_cmd)

    # 2. Dataset Loading & Split Integrity
    dataset_path = Path(args.dataset_dir)
    logger.info("Loading dataset from: %s", dataset_path)
    loader = DatasetLoader(dataset_path)
    train_records, val_records, test_records = loader.load_all_splits()

    train_stats = get_split_statistics(train_records)
    val_stats = get_split_statistics(val_records)
    test_stats = get_split_statistics(test_records)

    print("\n" + "-" * 75)
    print(f"Train samples:      {train_stats['total']} ({train_stats['genuine']} genuine, {train_stats['forged']} forged - {train_stats['forged_pct']}%)")
    print(f"Validation samples: {val_stats['total']} ({val_stats['genuine']} genuine, {val_stats['forged']} forged - {val_stats['forged_pct']}%)")
    print(f"Test samples:       {test_stats['total']} ({test_stats['genuine']} genuine, {test_stats['forged']} forged - {test_stats['forged_pct']}%)")
    print("-" * 75)

    # 3. Feature Extraction
    cache_path = Path(args.cache_dir)
    ocr_pipeline = OCRPipeline(tesseract_cmd=tesseract_cmd)
    extractor = FeatureExtractor(ocr_pipeline=ocr_pipeline)

    x_train, y_train = extract_features_for_split(train_records, "train", extractor, cache_path)
    x_val, y_val = extract_features_for_split(val_records, "val", extractor, cache_path)
    x_test, y_test = extract_features_for_split(test_records, "test", extractor, cache_path)

    # 4. Feature Scaling (Fitted STRICTLY on Train split only)
    logger.info("Fitting StandardScaler on Train set only (zero data leakage)...")
    trainer = ModelTrainer(random_state=args.seed)
    x_train_scaled = trainer.fit_scaler(x_train)
    x_val_scaled = trainer.transform_features(x_val)
    x_test_scaled = trainer.transform_features(x_test)

    # 5. Candidate Model Training & Validation Comparison
    print("\n" + "=" * 75)
    print("                    VALIDATION MODEL SELECTION")
    print("   Primary Criterion: Forged-Class F1 (label=1)")
    print("   Secondary Criteria: Forged-Class Recall, ROC-AUC")
    print("=" * 75)

    best_name, best_model, val_metrics = trainer.evaluate_candidates(
        x_train_scaled, y_train, x_val_scaled, y_val
    )

    print("\nCandidate Model Validation Comparison:")
    print("-" * 75)
    print(f"{'Model':<22} | {'Val Acc':<8} | {'Forged F1':<10} | {'Forged Rec':<10} | {'ROC-AUC':<8}")
    print("-" * 75)
    for name, m in val_metrics.items():
        prefix = "-> [BEST] " if name == best_name else "   "
        print(
            f"{prefix + name:<22} | {m.accuracy:<8.4f} | {m.f1_forged:<10.4f} | {m.recall_forged:<10.4f} | {m.roc_auc:<8.4f}"
        )
    print("-" * 75)
    print(f"Selected Model: {best_name}")

    # 6. Final Evaluation on Untouched Test Set
    print("\n" + "=" * 75)
    print("               FINAL UNBIASED EVALUATION ON TEST SET")
    print("=" * 75)

    y_test_pred = best_model.predict(x_test_scaled)
    if hasattr(best_model, "predict_proba"):
        y_test_prob = best_model.predict_proba(x_test_scaled)[:, 1]
    elif hasattr(best_model, "decision_function"):
        df = best_model.decision_function(x_test_scaled)
        y_test_prob = 1.0 / (1.0 + np.exp(-df))
    else:
        y_test_prob = None

    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_prob)

    print(f"Test Accuracy:                {test_metrics.accuracy:.4f}")
    print(f"Test Forged Precision:        {test_metrics.precision_forged:.4f}")
    print(f"Test Forged Recall:           {test_metrics.recall_forged:.4f}")
    print(f"Test Forged F1-Score:         {test_metrics.f1_forged:.4f}")
    print(f"Test ROC-AUC:                 {test_metrics.roc_auc:.4f}")
    print(f"Test Macro F1:                {test_metrics.f1_macro:.4f}")
    print("\nConfusion Matrix (Test Split):")
    print(f"  [TN={test_metrics.confusion_matrix[0][0]:<4}  FP={test_metrics.confusion_matrix[0][1]:<4}] (Genuine)")
    print(f"  [FN={test_metrics.confusion_matrix[1][0]:<4}  TP={test_metrics.confusion_matrix[1][1]:<4}] (Forged)")

    # 7. Save Artifacts & Reports
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)

    all_metrics_report = {
        "selected_model": best_name,
        "train_samples": train_stats["total"],
        "val_samples": val_stats["total"],
        "test_samples": test_stats["total"],
        "validation_comparison": {k: v.to_dict() for k, v in val_metrics.items()},
        "test_metrics": test_metrics.to_dict(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    save_pipeline(
        model=best_model,
        scaler=trainer.scaler,
        feature_names=FEATURE_NAMES,
        model_name=best_name,
        metrics=all_metrics_report,
        models_dir=models_dir,
    )

    # Save outputs/metrics.json
    metrics_json_path = outputs_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics_report, f, indent=2)
    logger.info("Saved test metrics to %s", metrics_json_path.name)

    # Save classification report text
    from sklearn.metrics import classification_report
    report_str = classification_report(
        y_test, y_test_pred, target_names=["Genuine (0)", "Forged (1)"], digits=4
    )
    with open(outputs_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Model: {best_name}\n")
        f.write(f"Timestamp: {all_metrics_report['timestamp']}\n\n")
        f.write(report_str)
        f.write("\n\nConfusion Matrix:\n")
        f.write(f"TN: {test_metrics.confusion_matrix[0][0]}, FP: {test_metrics.confusion_matrix[0][1]}\n")
        f.write(f"FN: {test_metrics.confusion_matrix[1][0]}, TP: {test_metrics.confusion_matrix[1][1]}\n")

    # Save visual evaluation plots
    save_evaluation_plots(
        y_test=y_test,
        y_pred=y_test_pred,
        y_prob=y_test_prob,
        model=best_model,
        feature_names=FEATURE_NAMES,
        outputs_dir=outputs_dir,
    )

    print("\n" + "=" * 75)
    print("          TRAINING AND EVALUATION COMPLETE SUCCESSFULLY")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

"""Phase 2 & Phase 3: Systematic Classical Model & Threshold Exploration.

Experiments with:
1. Regularization & Multicollinearity mitigation (L1 Lasso, PCA, Feature Selection)
2. Hyperparameter tuning across LogisticRegression, RandomForest, ExtraTrees, SVM, HistGradientBoosting
3. Cost-sensitive / Balanced class weighting
4. Decision threshold optimization on Validation probabilities (sweeping 0.05 to 0.95)

CRITICAL RULE:
Operates STRICTLY on Train split for fitting and Validation split for evaluation.
Test split is NEVER accessed or evaluated.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from src.feature_extraction import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("classical_experiments")


def load_cached_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and validation splits."""
    train_cache = CACHE_DIR / "features_train.npz"
    val_cache = CACHE_DIR / "features_val.npz"

    train_data = np.load(train_cache)
    val_data = np.load(val_cache)

    return train_data["x"], train_data["y"], val_data["x"], val_data["y"]


def evaluate_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """Calculate metrics at a specific decision threshold."""
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


def find_optimal_validation_threshold(y_val: np.ndarray, y_val_prob: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find threshold that maximizes validation Forged F1 score.

    Returns:
        (metrics_at_default_0_5, metrics_at_best_threshold)
    """
    default_metrics = evaluate_at_threshold(y_val, y_val_prob, threshold=0.50)

    best_metrics = default_metrics
    best_key = (default_metrics["f1_forged"], default_metrics["recall_forged"], default_metrics["roc_auc"])

    # Sweep thresholds from 0.05 to 0.95 in 0.01 increments
    for th in np.arange(0.05, 0.95, 0.01):
        m = evaluate_at_threshold(y_val, y_val_prob, threshold=float(th))
        key = (m["f1_forged"], m["recall_forged"], m["roc_auc"])
        if key > best_key:
            best_key = key
            best_metrics = m

    return default_metrics, best_metrics


def get_candidate_configurations() -> List[Tuple[str, Any]]:
    """Define candidate model architectures and preprocessing pipelines."""
    configs = []

    # 1. Baseline Logistic Regression (StandardScaler)
    configs.append((
        "LogisticRegression (L2, C=1.0, Balanced)",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)),
        ]),
    ))

    # 2. L1 Sparse Logistic Regression (eliminates redundant/collinear features)
    for c in [0.05, 0.1, 0.5, 1.0, 2.0]:
        configs.append((
            f"LogisticRegression (L1 Lasso, C={c}, Balanced)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", C=c, random_state=42)),
            ]),
        ))

    # 3. Logistic Regression with PCA (collapses 65 collinear dimensions)
    for n_comp in [5, 10, 15]:
        configs.append((
            f"LogisticRegression + PCA(n={n_comp})",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp, random_state=42)),
                ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)),
            ]),
        ))

    # 4. Logistic Regression with SelectKBest (top 10 signal features)
    configs.append((
        "LogisticRegression + SelectKBest(k=10)",
        Pipeline([
            ("scaler", StandardScaler()),
            ("select", SelectKBest(f_classif, k=10)),
            ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000, random_state=42)),
        ]),
    ))

    # 5. Random Forest Variations
    for depth in [4, 6, 8, None]:
        for cw in ["balanced", "balanced_subsample"]:
            configs.append((
                f"RandomForest (depth={depth}, cw={cw})",
                Pipeline([
                    ("clf", RandomForestClassifier(
                        n_estimators=150,
                        max_depth=depth,
                        min_samples_leaf=3,
                        class_weight=cw,
                        random_state=42,
                        n_jobs=-1,
                    )),
                ]),
            ))

    # 6. ExtraTrees Classifier
    for depth in [4, 6, 8]:
        configs.append((
            f"ExtraTrees (depth={depth}, cw=balanced)",
            Pipeline([
                ("clf", ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=depth,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )),
            ]),
        ))

    # 7. Support Vector Machines
    for c_val in [0.1, 0.5, 1.0, 5.0]:
        configs.append((
            f"SVM Linear (C={c_val}, Balanced)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", C=c_val, class_weight="balanced", probability=True, random_state=42)),
            ]),
        ))
        configs.append((
            f"SVM RBF (C={c_val}, Balanced)",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=c_val, gamma="scale", class_weight="balanced", probability=True, random_state=42)),
            ]),
        ))

    # 8. HistGradientBoosting (tuned regularization)
    for l2_reg in [0.0, 1.0, 5.0, 10.0]:
        for min_leaf in [10, 20, 40]:
            configs.append((
                f"HistGradientBoosting (l2={l2_reg}, min_leaf={min_leaf})",
                Pipeline([
                    ("clf", HistGradientBoostingClassifier(
                        class_weight="balanced",
                        l2_regularization=l2_reg,
                        min_samples_leaf=min_leaf,
                        max_iter=100,
                        random_state=42,
                    )),
                ]),
            ))

    # 9. AdaBoost
    configs.append((
        "AdaBoostClassifier (n=50, lr=0.5)",
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", AdaBoostClassifier(n_estimators=50, learning_rate=0.5, random_state=42)),
        ]),
    ))

    return configs


def run_experiments():
    print("\n" + "=" * 90)
    print("      PHASE 2 & PHASE 3: CLASSICAL MODEL COMPARISON & THRESHOLD OPTIMIZATION")
    print("      (Conducted STRICTLY on Train & Validation Splits — Zero Test Leakage)")
    print("=" * 90)

    x_train, y_train, x_val, y_val = load_cached_data()

    configs = get_candidate_configurations()
    logger.info("Evaluating %d model configurations on Validation set...", len(configs))

    results = []

    for name, pipeline in configs:
        try:
            # Fit on training data only
            pipeline.fit(x_train, y_train)

            # Predict probabilities on validation data only
            if hasattr(pipeline, "predict_proba"):
                y_val_prob = pipeline.predict_proba(x_val)[:, 1]
            elif hasattr(pipeline, "decision_function"):
                df = pipeline.decision_function(x_val)
                y_val_prob = 1.0 / (1.0 + np.exp(-df))
            else:
                y_val_pred = pipeline.predict(x_val)
                y_val_prob = y_val_pred.astype(float)

            # Default (0.50) vs Optimal Validation Threshold
            m_default, m_opt = find_optimal_validation_threshold(y_val, y_val_prob)

            results.append({
                "model_name": name,
                "default": m_default,
                "optimized": m_opt,
            })

        except Exception as e:
            logger.warning("Error evaluating %s: %s", name, e)

    # Sort results by optimized validation Forged F1
    results.sort(key=lambda r: (r["optimized"]["f1_forged"], r["optimized"]["recall_forged"], r["optimized"]["roc_auc"]), reverse=True)

    print("\n" + "=" * 95)
    print("                       TOP 15 MODELS ON VALIDATION SET")
    print("   Ranked by Validation Forged-Class F1 (Primary Selection Metric)")
    print("=" * 95)
    print(f"{'Model Configuration':<36} | {'Opt Thresh':<10} | {'Val Acc':<8} | {'Forged Prec':<11} | {'Forged Rec':<10} | {'Forged F1':<9} | {'ROC-AUC':<8}")
    print("-" * 105)

    for i, r in enumerate(results[:15], 1):
        opt = r["optimized"]
        prefix = f"{i:>2}. "
        print(
            f"{prefix + r['model_name']:<36} | {opt['threshold']:<10.2f} | {opt['accuracy']:<8.4f} | "
            f"{opt['precision_forged']:<11.4f} | {opt['recall_forged']:<10.4f} | {opt['f1_forged']:<9.4f} | {opt['roc_auc']:<8.4f}"
        )

    print("-" * 105)

    # Show Threshold Analysis for the Top 5 Models
    print("\n" + "=" * 95)
    print("          PHASE 3: THRESHOLD ANALYSIS — BEFORE (0.50) vs AFTER OPTIMIZATION")
    print("=" * 95)
    print(f"{'Model Configuration':<36} | {'Thresh':<7} | {'Val Acc':<8} | {'Forged Prec':<11} | {'Forged Rec':<10} | {'Forged F1':<9}")
    print("-" * 95)

    for r in results[:5]:
        d = r["default"]
        opt = r["optimized"]
        print(f"{r['model_name']:<36} | Def:0.50| {d['accuracy']:<8.4f} | {d['precision_forged']:<11.4f} | {d['recall_forged']:<10.4f} | {d['f1_forged']:<9.4f}")
        print(f"{'':<36} | Opt:{opt['threshold']:<4.2f}| {opt['accuracy']:<8.4f} | {opt['precision_forged']:<11.4f} | {opt['recall_forged']:<10.4f} | {opt['f1_forged']:<9.4f}")
        print("-" * 95)

    # Save complete experiments JSON
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    exp_file = OUTPUTS_DIR / "classical_experiments_report.json"
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved classical experiments report to %s", exp_file)


if __name__ == "__main__":
    run_experiments()

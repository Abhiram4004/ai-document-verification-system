"""Phase 1: Feature Matrix Diagnosis and Signal Analysis.

Analyzes the 37 deterministic features across Train and Validation splits:
- Class-conditional distribution statistics (mean, std, median, IQR)
- Univariate discriminative power (Welch's t-test, Mann-Whitney U, Cohen's d, univariate ROC-AUC)
- Mutual information with forgery label
- Feature collinearity and redundancy analysis
- OCR features vs Image features comparison
NOTE: Uses ONLY Train and Validation splits. Test split is NEVER accessed.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

from src.feature_extraction import FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("diagnose")


def load_cached_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load train and validation feature matrices and labels."""
    train_cache = CACHE_DIR / "features_train.npz"
    val_cache = CACHE_DIR / "features_val.npz"

    if not train_cache.is_file() or not val_cache.is_file():
        raise FileNotFoundError(
            f"Feature caches missing in {CACHE_DIR}. Ensure features_train.npz and features_val.npz exist."
        )

    train_data = np.load(train_cache)
    val_data = np.load(val_cache)

    x_train, y_train = train_data["x"], train_data["y"]
    x_val, y_val = val_data["x"], val_data["y"]

    logger.info("Loaded Train: %s, Val: %s", x_train.shape, x_val.shape)
    return x_train, y_train, x_val, y_val


def compute_univariate_feature_stats(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> List[Dict[str, Any]]:
    """Compute statistical discrimination metrics for each feature on training set."""
    is_genuine = (y == 0)
    is_forged = (y == 1)

    # Compute mutual information
    mi_scores = mutual_info_classif(x, y, random_state=42, n_neighbors=5)

    feature_stats = []

    for i, name in enumerate(feature_names):
        vals_gen = x[is_genuine, i]
        vals_forg = x[is_forged, i]

        mean_gen, std_gen = float(np.mean(vals_gen)), float(np.std(vals_gen))
        mean_forg, std_forg = float(np.mean(vals_forg)), float(np.std(vals_forg))

        med_gen = float(np.median(vals_gen))
        med_forg = float(np.median(vals_forg))

        # Welch's t-test (two-sample unequal variance)
        try:
            t_stat, p_welch = stats.ttest_ind(vals_gen, vals_forg, equal_var=False)
            p_welch = float(p_welch)
        except Exception:
            p_welch = 1.0

        # Mann-Whitney U test (non-parametric rank sum)
        try:
            u_stat, p_mwu = stats.mannwhitneyu(vals_gen, vals_forg, alternative="two-sided")
            p_mwu = float(p_mwu)
        except Exception:
            p_mwu = 1.0

        # Cohen's d effect size: (mean_forged - mean_genuine) / pooled_std
        pooled_std = np.sqrt(((len(vals_gen) - 1) * (std_gen ** 2) + (len(vals_forg) - 1) * (std_forg ** 2)) / (len(x) - 2))
        if pooled_std > 1e-6:
            cohens_d = float((mean_forg - mean_gen) / pooled_std)
        else:
            cohens_d = 0.0

        # Univariate ROC-AUC
        try:
            if len(np.unique(x[:, i])) > 1:
                auc_raw = float(roc_auc_score(y, x[:, i]))
                # Effective directional AUC (distance from 0.5)
                auc_dir = auc_raw if auc_raw >= 0.5 else (1.0 - auc_raw)
            else:
                auc_raw = 0.5
                auc_dir = 0.5
        except Exception:
            auc_raw = 0.5
            auc_dir = 0.5

        # Feature type category
        if name.startswith("ocr_"):
            category = "OCR"
        elif name.startswith("geom_"):
            category = "Geometry"
        elif name.startswith("gray_"):
            category = "Grayscale/Contrast"
        elif name.startswith("color_"):
            category = "Color"
        else:
            category = "Sharpness/Edge"

        feature_stats.append({
            "name": name,
            "category": category,
            "mean_genuine": round(mean_gen, 4),
            "std_genuine": round(std_gen, 4),
            "median_genuine": round(med_gen, 4),
            "mean_forged": round(mean_forg, 4),
            "std_forged": round(std_forg, 4),
            "median_forged": round(med_forg, 4),
            "p_welch": round(p_welch, 6),
            "p_mann_whitney": round(p_mwu, 6),
            "cohens_d": round(cohens_d, 4),
            "univariate_auc_raw": round(auc_raw, 4),
            "univariate_auc_abs": round(auc_dir, 4),
            "mutual_info": round(float(mi_scores[i]), 5),
            "is_significant_005": bool(p_mwu < 0.05 or p_welch < 0.05),
        })

    return feature_stats


def analyze_correlations(x_train: np.ndarray, feature_names: List[str], threshold: float = 0.85) -> List[Dict[str, Any]]:
    """Identify highly collinear pairs of features on the training data."""
    df = pd.DataFrame(x_train, columns=feature_names)
    corr_matrix = df.corr(method="spearman").abs()

    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            r = float(corr_matrix.iloc[i, j])
            if r >= threshold:
                high_corr_pairs.append({
                    "feature_a": feature_names[i],
                    "feature_b": feature_names[j],
                    "spearman_rho": round(r, 4),
                })

    high_corr_pairs.sort(key=lambda item: item["spearman_rho"], reverse=True)
    return high_corr_pairs


def run_diagnosis():
    print("\n" + "=" * 80)
    print("      PHASE 1: BASELINE FEATURE DIAGNOSIS & SIGNAL ANALYSIS")
    print("      (Conducted STRICTLY on Train & Validation Data — Zero Test Leakage)")
    print("=" * 80)

    x_train, y_train, x_val, y_val = load_cached_data()

    print(f"\n[Split Sizes]")
    print(f"  Train:      {len(y_train)} samples ({np.sum(y_train == 0)} Genuine, {np.sum(y_train == 1)} Forged — {np.mean(y_train)*100:.1f}% Forged)")
    print(f"  Validation: {len(y_val)} samples ({np.sum(y_val == 0)} Genuine, {np.sum(y_val == 1)} Forged — {np.mean(y_val)*100:.1f}% Forged)")

    # 1. Univariate Statistical Signal Analysis
    stats_list = compute_univariate_feature_stats(x_train, y_train, FEATURE_NAMES)

    # Sort features by univariate discriminative power (absolute AUC distance from 0.5)
    stats_list.sort(key=lambda x: abs(x["univariate_auc_raw"] - 0.5), reverse=True)

    print("\n" + "-" * 80)
    print(f"{'Feature Name':<24} | {'Category':<16} | {'AUC':<6} | {'Cohen d':<8} | {'p-MWU':<8} | {'MI':<7} | {'Sig?'}")
    print("-" * 80)
    for s in stats_list:
        sig_str = "[YES]" if s["is_significant_005"] else " no "
        print(
            f"{s['name']:<24} | {s['category']:<16} | {s['univariate_auc_raw']:<6.3f} | "
            f"{s['cohens_d']:<+8.3f} | {s['p_mann_whitney']:<8.4f} | {s['mutual_info']:<7.4f} | {sig_str}"
        )
    print("-" * 80)

    # 2. Group Summary: OCR vs Visual features
    categories = sorted(list(set(s["category"] for s in stats_list)))
    print("\n[Signal Summary by Feature Category (Train Split)]")
    print(f"{'Category':<20} | {'Count':<5} | {'Significant (p<0.05)':<20} | {'Avg |AUC-0.5|':<14} | {'Max AUC':<8}")
    print("-" * 75)
    category_summary = {}
    for cat in categories:
        cat_feats = [s for s in stats_list if s["category"] == cat]
        sig_cnt = sum(1 for s in cat_feats if s["is_significant_005"])
        avg_auc_delta = float(np.mean([abs(s["univariate_auc_raw"] - 0.5) for s in cat_feats]))
        max_auc = max(s["univariate_auc_abs"] for s in cat_feats)
        category_summary[cat] = {
            "count": len(cat_feats),
            "significant_count": sig_cnt,
            "avg_auc_delta": round(avg_auc_delta, 4),
            "max_auc": round(max_auc, 4),
        }
        print(f"{cat:<20} | {len(cat_feats):<5} | {sig_cnt}/{len(cat_feats)} ({sig_cnt/len(cat_feats)*100:.0f}%)"
              f"{'':<10} | {avg_auc_delta:<14.4f} | {max_auc:<8.4f}")
    print("-" * 75)

    # 3. Collinearity / Redundancy Analysis
    collinear_pairs = analyze_correlations(x_train, FEATURE_NAMES, threshold=0.80)
    print(f"\n[High Correlation Pairs (|Spearman rho| >= 0.80)] ({len(collinear_pairs)} pairs found)")
    print("-" * 75)
    for p in collinear_pairs[:12]:
        print(f"  {p['feature_a']:<24} <---> {p['feature_b']:<24} (rho = {p['spearman_rho']:.4f})")
    if len(collinear_pairs) > 12:
        print(f"  ... and {len(collinear_pairs) - 12} additional collinear pairs.")

    # 4. Check validation stability of top signals
    top_train_features = [s["name"] for s in stats_list[:8]]
    val_stats = compute_univariate_feature_stats(x_val, y_val, top_train_features)
    val_auc_map = {s["name"]: s["univariate_auc_raw"] for s in val_stats}

    print("\n[Cross-Split Signal Stability (Train vs Validation)]")
    print(f"{'Feature Name':<24} | {'Train AUC':<10} | {'Val AUC':<10} | {'Stable Direction?'}")
    print("-" * 65)
    stability_data = []
    for s in stats_list[:8]:
        fname = s["name"]
        tr_auc = s["univariate_auc_raw"]
        v_auc = val_auc_map.get(fname, 0.5)
        # Consistent if both > 0.5 or both < 0.5
        consistent = (tr_auc >= 0.5 and v_auc >= 0.5) or (tr_auc < 0.5 and v_auc < 0.5)
        st_mark = "YES (Consistent)" if consistent else "NO  (Inverted!)"
        stability_data.append({
            "feature": fname,
            "train_auc": round(tr_auc, 4),
            "val_auc": round(v_auc, 4),
            "consistent": bool(consistent),
        })
        print(f"{fname:<24} | {tr_auc:<10.4f} | {v_auc:<10.4f} | {st_mark}")
    print("-" * 65)

    # Save complete diagnostic JSON
    diagnosis_output = {
        "dataset_splits": {
            "train_total": len(y_train),
            "train_genuine": int(np.sum(y_train == 0)),
            "train_forged": int(np.sum(y_train == 1)),
            "val_total": len(y_val),
            "val_genuine": int(np.sum(y_val == 0)),
            "val_forged": int(np.sum(y_val == 1)),
        },
        "feature_statistics": stats_list,
        "category_summary": category_summary,
        "collinear_pairs": collinear_pairs,
        "train_val_stability": stability_data,
    }

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUTPUTS_DIR / "feature_diagnosis_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(diagnosis_output, f, indent=2)
    logger.info("Saved feature diagnosis report to %s", report_file)


if __name__ == "__main__":
    run_diagnosis()

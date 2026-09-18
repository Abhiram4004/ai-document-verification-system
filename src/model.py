"""Machine learning classification pipeline, model comparison, and serialization."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
except ImportError:
    pass


@dataclass
class EvaluationMetrics:
    """Standard evaluation metrics for binary classification."""
    accuracy: float
    precision_forged: float  # Label 1 (Forged)
    recall_forged: float     # Label 1 (Forged)
    f1_forged: float         # Primary selection metric
    roc_auc: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> EvaluationMetrics:
    """Compute comprehensive classification metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_1 = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec_1 = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    f1_1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))

    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.0
    else:
        auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvaluationMetrics(
        accuracy=round(acc, 4),
        precision_forged=round(prec_1, 4),
        recall_forged=round(rec_1, 4),
        f1_forged=round(f1_1, 4),
        roc_auc=round(auc, 4),
        precision_macro=round(prec_macro, 4),
        recall_macro=round(rec_macro, 4),
        f1_macro=round(f1_macro, 4),
        confusion_matrix=cm,
    )


class ModelTrainer:
    """Trains and compares candidate classifiers on validation split."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()

    def get_candidate_models(self) -> Dict[str, Any]:
        """Instantiate candidate classifiers with balanced class weighting."""
        return {
            "LogisticRegression": LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "SVM_RBF": SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced",
                probability=True,
                random_state=self.random_state,
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=100,
                random_state=self.random_state,
            ),
        }

    def fit_scaler(self, x_train: np.ndarray) -> np.ndarray:
        """Fit feature scaler strictly on the train split only."""
        return self.scaler.fit_transform(x_train)

    def transform_features(self, x: np.ndarray) -> np.ndarray:
        """Transform features using pre-fitted scaler."""
        return self.scaler.transform(x)

    def evaluate_candidates(
        self,
        x_train_scaled: np.ndarray,
        y_train: np.ndarray,
        x_val_scaled: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[str, Any, Dict[str, EvaluationMetrics]]:
        """Fit candidates on train set and evaluate strictly on validation set.

        Selection Criterion:
            Primary: Forged-class F1 score (label=1)
            Tie-breakers: Forged-class Recall, then ROC-AUC.
        """
        candidates = self.get_candidate_models()
        val_metrics: Dict[str, EvaluationMetrics] = {}

        best_model_name = ""
        best_key = (-1.0, -1.0, -1.0)  # (f1_forged, recall_forged, roc_auc)
        best_model = None

        for name, model in candidates.items():
            logger.info("Training candidate model: %s", name)
            model.fit(x_train_scaled, y_train)

            y_val_pred = model.predict(x_val_scaled)
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(x_val_scaled)[:, 1]
            elif hasattr(model, "decision_function"):
                df = model.decision_function(x_val_scaled)
                y_val_prob = 1.0 / (1.0 + np.exp(-df))
            else:
                y_val_prob = None

            metrics = calculate_metrics(y_val, y_val_pred, y_val_prob)
            val_metrics[name] = metrics
            logger.info(
                "[%s] Val F1(forged): %.4f | Val Recall(forged): %.4f | Val ROC-AUC: %.4f | Val Acc: %.4f",
                name,
                metrics.f1_forged,
                metrics.recall_forged,
                metrics.roc_auc,
                metrics.accuracy,
            )

            candidate_key = (metrics.f1_forged, metrics.recall_forged, metrics.roc_auc)
            if candidate_key > best_key:
                best_key = candidate_key
                best_model_name = name
                best_model = model

        logger.info(
            "Selected best model '%s' with validation Forged F1: %.4f",
            best_model_name,
            best_key[0],
        )
        return best_model_name, best_model, val_metrics


def get_model_interpretation(model: Any, feature_names: List[str]) -> Dict[str, Any]:
    """Extract valid model interpretation without misrepresenting non-applicable attributes.

    Returns:
        Dict with interpretation type and ordered (feature, value) items.
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {
            "type": "feature_importance",
            "title": "Random Forest Feature Importances",
            "values": [{"feature": f, "importance": round(float(v), 5)} for f, v in sorted_pairs],
        }

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        sorted_pairs = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        return {
            "type": "coefficients",
            "title": "Logistic Regression Model Coefficients",
            "values": [{"feature": f, "coefficient": round(float(v), 5)} for f, v in sorted_pairs],
        }

    return {
        "type": "none",
        "title": "Model Interpretation Not Supported",
        "message": f"Model '{type(model).__name__}' does not expose direct linear coefficients or tree feature importances.",
        "values": [],
    }


def save_pipeline(
    model: Any,
    scaler: Any,
    feature_names: List[str],
    model_name: str,
    metrics: Dict[str, Any],
    models_dir: Path,
) -> Tuple[Path, Path, Path]:
    """Serialize model artifacts and metadata."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "document_verifier.joblib"
    pipeline_path = models_dir / "feature_pipeline.joblib"
    metadata_path = models_dir / "metadata.json"

    # Save classifier
    joblib.dump(model, model_path)

    # Save feature pipeline (scaler + feature names)
    feature_pipeline = {
        "scaler": scaler,
        "feature_names": feature_names,
        "model_name": model_name,
    }
    joblib.dump(feature_pipeline, pipeline_path)

    # Save metadata
    import datetime
    metadata = {
        "model_name": model_name,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "num_features": len(feature_names),
        "feature_names": feature_names,
        "metrics": metrics,
        "probability_notice": "Model probability represents mathematical confidence and is NOT a legal or certified authenticity score.",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved model artifacts to %s", models_dir)
    return model_path, pipeline_path, metadata_path


def load_pipeline(models_dir: Path) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    """Deserialize model, scaler, feature names, and metadata."""
    models_dir = Path(models_dir)
    model_path = models_dir / "document_verifier.joblib"
    pipeline_path = models_dir / "feature_pipeline.joblib"
    metadata_path = models_dir / "metadata.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact missing: {model_path}")
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Pipeline artifact missing: {pipeline_path}")

    model = joblib.load(model_path)
    pipeline_data = joblib.load(pipeline_path)
    scaler = pipeline_data["scaler"]
    feature_names = pipeline_data["feature_names"]

    metadata = {}
    if metadata_path.is_file():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return model, scaler, feature_names, metadata

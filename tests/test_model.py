"""Unit tests for model training, metrics calculation, and serialization."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.feature_extraction import FEATURE_NAMES
from src.model import (
    ModelTrainer,
    calculate_metrics,
    get_model_interpretation,
    load_pipeline,
    save_pipeline,
)


def test_calculate_metrics():
    """Test classification metrics computation."""
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 0, 0])  # 1 TP, 0 FP, 3 TN, 1 FN
    y_prob = np.array([0.1, 0.2, 0.9, 0.4, 0.1])

    m = calculate_metrics(y_true, y_pred, y_prob)

    assert m.accuracy == 0.8
    assert m.recall_forged == 0.5
    assert m.precision_forged == 1.0
    assert m.f1_forged == pytest.approx(0.6667, 0.01)
    assert m.confusion_matrix == [[3, 0], [1, 1]]


def test_model_training_and_serialization():
    """Test training, candidate selection, pipeline serialization and deserialization."""
    # Synthetic dataset
    np.random.seed(42)
    n_samples = 40
    n_features = len(FEATURE_NAMES)
    x_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

    x_val = np.random.randn(20, n_features).astype(np.float32)
    y_val = np.random.choice([0, 1], size=20, p=[0.8, 0.2])

    trainer = ModelTrainer(random_state=42)
    x_train_scaled = trainer.fit_scaler(x_train)
    x_val_scaled = trainer.transform_features(x_val)

    best_name, best_model, val_metrics = trainer.evaluate_candidates(
        x_train_scaled, y_train, x_val_scaled, y_val
    )

    assert best_name in trainer.get_candidate_models()
    assert best_model is not None
    assert len(val_metrics) == 4

    # Test saving and loading
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir)
        save_pipeline(
            model=best_model,
            scaler=trainer.scaler,
            feature_names=FEATURE_NAMES,
            model_name=best_name,
            metrics={"test": 123},
            models_dir=models_dir,
        )

        loaded_model, loaded_scaler, loaded_feats, loaded_meta = load_pipeline(models_dir)

        assert loaded_feats == FEATURE_NAMES
        assert loaded_meta["model_name"] == best_name

        # Ensure predictions match
        pred1 = best_model.predict(x_val_scaled)
        pred2 = loaded_model.predict(loaded_scaler.transform(x_val))
        assert np.array_equal(pred1, pred2)

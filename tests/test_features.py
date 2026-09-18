"""Unit tests for feature extraction pipeline."""

import numpy as np
import pytest
from src.feature_extraction import FEATURE_NAMES, FeatureExtractor
from src.ocr import OCRPipeline


def test_feature_extraction_shape_and_cleanliness(sample_image):
    """Test feature vector dimension, completeness, and lack of NaN/Inf."""
    ocr_mock = OCRPipeline(use_cache=False)
    extractor = FeatureExtractor(ocr_pipeline=ocr_mock)

    vec, feat_dict, text = extractor.extract_features(
        sample_image,
        mock_ocr_text="SAMPLE RECEIPT 100",
    )

    # Dimension must exactly match FEATURE_NAMES
    assert len(vec) == len(FEATURE_NAMES)
    assert vec.ndim == 1

    # Check for NaN and Inf
    assert not np.isnan(vec).any(), "Feature vector contains NaN!"
    assert not np.isinf(vec).any(), "Feature vector contains Inf!"

    # Check dictionary keys
    for name in FEATURE_NAMES:
        assert name in feat_dict


def test_feature_determinism(sample_image):
    """Ensure identical inputs produce identical feature vectors."""
    ocr_mock = OCRPipeline(use_cache=False)
    extractor = FeatureExtractor(ocr_pipeline=ocr_mock)

    vec1, _, _ = extractor.extract_features(sample_image, mock_ocr_text="TEST 123")
    vec2, _, _ = extractor.extract_features(sample_image, mock_ocr_text="TEST 123")

    assert np.allclose(vec1, vec2, atol=1e-5)

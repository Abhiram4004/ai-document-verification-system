"""Unit tests for image preprocessing module."""

import numpy as np
import pytest
from PIL import Image
from src.preprocessing import ImagePreprocessor


def test_load_image(sample_image, sample_image_bytes):
    """Test loading images from various input representations."""
    pre = ImagePreprocessor()

    # From PIL Image
    arr1 = pre.load_image(sample_image)
    assert isinstance(arr1, np.ndarray)
    assert arr1.ndim == 3
    assert arr1.shape[2] == 3

    # From bytes
    arr2 = pre.load_image(sample_image_bytes)
    assert isinstance(arr2, np.ndarray)
    assert arr2.shape == arr1.shape

    # From numpy array
    arr3 = pre.load_image(arr1)
    assert np.array_equal(arr1, arr3)


def test_resize_keep_aspect(sample_image):
    """Test aspect-preserving resizing."""
    pre = ImagePreprocessor(target_max_dim=300)
    arr = pre.load_image(sample_image)  # Original is 400x600 (w=400, h=600)
    resized = pre.resize_keep_aspect(arr, max_dim=300)

    h, w = resized.shape[:2]
    assert max(h, w) == 300
    assert w == 200  # 400 * (300/600)


def test_grayscale_and_binarize(sample_image):
    """Test grayscale conversion and binarization."""
    pre = ImagePreprocessor()
    arr = pre.load_image(sample_image)
    gray = pre.to_grayscale(arr)

    assert gray.ndim == 2
    assert gray.shape[:2] == arr.shape[:2]
    assert gray.dtype == np.uint8

    binary = pre.binarize(gray)
    assert binary.ndim == 2
    # Binarized should only contain 0 and 255
    unique_vals = set(np.unique(binary))
    assert unique_vals.issubset({0, 255})

"""Unit tests for visual deep learning model and Grad-CAM interpretability."""

import io
import numpy as np
import pytest
from PIL import Image

try:
    import torch
    from src.visual_model import (
        DocumentForgeryResNet,
        compute_gradcam,
        get_visual_transforms,
        overlay_gradcam,
    )
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.verification import DocumentVerifier


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_document_forgery_resnet_architecture():
    """Test model structure, forward pass shape, and freezing mechanisms."""
    model = DocumentForgeryResNet(pretrained=False, dropout=0.3)
    model.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy_input)

    assert out.shape == (2, 2), f"Expected shape (2, 2), got {out.shape}"

    # Test backbone freeze
    model.freeze_backbone()
    for name, param in model.named_parameters():
        if "classifier" not in name:
            assert not param.requires_grad
        else:
            assert param.requires_grad

    # Test unfreeze upper layers
    model.unfreeze_upper_layers()
    for name, param in model.named_parameters():
        if "layer4" in name or "classifier" in name:
            assert param.requires_grad
        else:
            assert not param.requires_grad


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_gradcam_computation():
    """Test Grad-CAM activation map computation and bounds."""
    model = DocumentForgeryResNet(pretrained=False)
    model.eval()

    dummy_tensor = torch.randn(1, 3, 128, 128)
    cam = compute_gradcam(model, dummy_tensor, target_class=1)

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (128, 128)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_overlay_gradcam():
    """Test blending Grad-CAM map onto PIL image."""
    img = Image.new("RGB", (200, 300), color=(255, 255, 255))
    cam = np.random.rand(128, 128).astype(np.float32)

    blended = overlay_gradcam(img, cam, alpha=0.4)
    assert isinstance(blended, Image.Image)
    assert blended.size == (200, 300)
    assert blended.mode == "RGB"


def test_verifier_gradcam_integration():
    """Test DocumentVerifier integration with Grad-CAM."""
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    verifier = DocumentVerifier()

    if verifier.visual_model is not None:
        overlay, cam, prob = verifier.generate_gradcam_overlay(img)
        assert overlay is not None
        assert cam is not None
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0
        assert overlay.size == (100, 100)

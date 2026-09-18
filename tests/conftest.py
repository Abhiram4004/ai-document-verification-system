"""Test fixtures and mock utilities for AI Document Verification System."""

import io
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw


@pytest.fixture
def sample_image():
    """Create a synthetic PIL document image with text-like shapes."""
    img = Image.new("RGB", (400, 600), color=(250, 250, 245))
    draw = ImageDraw.Draw(img)
    # Draw simulated text lines
    for y in range(50, 500, 30):
        draw.rectangle([50, y, 350, y + 15], fill=(30, 30, 30))
    return img


@pytest.fixture
def sample_image_bytes(sample_image):
    """Return synthetic image as PNG bytes."""
    buf = io.BytesIO()
    sample_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def temp_dataset_dir(sample_image):
    """Create a small self-contained temporary dataset for unit testing data loading."""
    tmp_path = Path(tempfile.mkdtemp())
    try:
        # Create train and val folders
        train_dir = tmp_path / "train"
        val_dir = tmp_path / "val"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        val_dir.mkdir()
        test_dir.mkdir()

        # Save synthetic images
        sample_image.save(train_dir / "sample_0.png")
        sample_image.save(train_dir / "sample_1.png")
        sample_image.save(train_dir / "duplicate.png")
        sample_image.save(val_dir / "val_0.png")
        sample_image.save(test_dir / "test_0.png")

        # Create annotation files
        # Notice: duplicate.png is in train, and also referenced in val.txt to test leakage prevention
        with open(tmp_path / "train.txt", "w", encoding="utf-8") as f:
            f.write("image,digital,handwritten,forged,details\n")
            f.write("sample_0.png,1,0,0,0\n")
            f.write("sample_1.png,0,1,1,{'Software': 'paint'}\n")
            f.write("duplicate.png,0,0,0,0\n")

        with open(tmp_path / "val.txt", "w", encoding="utf-8") as f:
            f.write("image,digital,handwritten,forged,details\n")
            f.write("val_0.png,1,0,0,0\n")
            f.write("duplicate.png,0,0,0,0\n")  # Cross-split duplicate to be filtered

        with open(tmp_path / "test.txt", "w", encoding="utf-8") as f:
            f.write("image,digital,handwritten,forged,details\n")
            f.write("test_0.png,0,0,1,0\n")

        yield tmp_path
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

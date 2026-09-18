"""Unit tests for dataset loading, split integrity, and leakage prevention."""

from pathlib import Path
import pytest
from src.data_loader import DatasetLoader, get_split_statistics


def test_dataset_loader_splits(temp_dataset_dir):
    """Test that DatasetLoader loads all splits and prevents cross-split leakage."""
    loader = DatasetLoader(temp_dataset_dir)
    train_records, val_records, test_records = loader.load_all_splits()

    # Train should have 3 samples (sample_0, sample_1, duplicate)
    assert len(train_records) == 3
    train_filenames = {r.filename for r in train_records}
    assert "sample_0.png" in train_filenames
    assert "sample_1.png" in train_filenames
    assert "duplicate.png" in train_filenames

    # Validation should have exactly 1 sample because duplicate.png was pruned
    assert len(val_records) == 1
    assert val_records[0].filename == "val_0.png"

    # Test should have 1 sample
    assert len(test_records) == 1
    assert test_records[0].filename == "test_0.png"

    # Ensure zero overlap
    val_filenames = {r.filename for r in val_records}
    test_filenames = {r.filename for r in test_records}
    assert not (train_filenames & val_filenames)
    assert not (train_filenames & test_filenames)
    assert not (val_filenames & test_filenames)


def test_split_statistics():
    """Test statistics aggregation helper."""
    from src.data_loader import DocumentRecord

    records = [
        DocumentRecord("a.png", Path("a.png"), 0),
        DocumentRecord("b.png", Path("b.png"), 0),
        DocumentRecord("c.png", Path("c.png"), 1),
    ]
    stats = get_split_statistics(records)
    assert stats["total"] == 3
    assert stats["genuine"] == 2
    assert stats["forged"] == 1
    assert stats["forged_pct"] == 33.33

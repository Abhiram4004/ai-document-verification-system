"""Dataset loading and split integrity module for Find it Again! dataset."""

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Represents a verified document sample."""
    filename: str
    image_path: Path
    label: int  # 0 = genuine, 1 = forged
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_forged(self) -> bool:
        return self.label == 1


class DatasetLoader:
    """Loads and validates dataset splits with strict leakage prevention."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = Path(dataset_root).resolve()
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.dataset_root}")

    def load_split(
        self,
        split_name: str,
        exclude_filenames: Optional[Set[str]] = None,
    ) -> List[DocumentRecord]:
        """Load and validate a dataset split (train, val, or test).

        Args:
            split_name: Name of split ('train', 'val', or 'test')
            exclude_filenames: Filenames to exclude (e.g., cross-split duplicates)

        Returns:
            List of verified DocumentRecord instances
        """
        exclude_filenames = exclude_filenames or set()
        txt_path = self.dataset_root / f"{split_name}.txt"
        img_dir = self.dataset_root / split_name

        if not txt_path.is_file():
            raise FileNotFoundError(f"Annotation file missing for split '{split_name}': {txt_path}")
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory missing for split '{split_name}': {img_dir}")

        records: List[DocumentRecord] = []
        seen_filenames: Set[str] = set()

        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)

            for line_idx, row in enumerate(reader, start=2):
                if not row or len(row) < 4:
                    logger.warning("Skipping malformed row %d in %s: %s", line_idx, txt_path.name, row)
                    continue

                filename = row[0].strip()
                if not filename:
                    continue

                # Filter excluded records (e.g. cross-split duplicate)
                if filename in exclude_filenames:
                    logger.info("Excluding known cross-split duplicate '%s' from %s", filename, split_name)
                    continue

                # Intra-split duplicate check
                if filename in seen_filenames:
                    logger.warning("Duplicate filename '%s' found in %s; skipping duplicate", filename, split_name)
                    continue

                # Parse target label
                try:
                    label = int(row[3].strip())
                    if label not in (0, 1):
                        logger.warning("Invalid label '%s' in row %d; skipping", row[3], line_idx)
                        continue
                except ValueError:
                    logger.warning("Cannot parse label '%s' in row %d; skipping", row[3], line_idx)
                    continue

                # Validate image file exists on disk
                image_path = img_dir / filename
                if not image_path.is_file():
                    logger.warning("Image file missing on disk: %s (row %d); skipping", image_path, line_idx)
                    continue

                # Preserve metadata for analysis only, never for model features
                metadata = {
                    "digital_annotation": row[1].strip() if len(row) > 1 else "",
                    "handwritten_annotation": row[2].strip() if len(row) > 2 else "",
                    "raw_forgery_details": row[4].strip() if len(row) > 4 else "",
                }

                records.append(
                    DocumentRecord(
                        filename=filename,
                        image_path=image_path,
                        label=label,
                        metadata=metadata,
                    )
                )
                seen_filenames.add(filename)

        return records

    def load_all_splits(self) -> Tuple[List[DocumentRecord], List[DocumentRecord], List[DocumentRecord]]:
        """Loads train, val, and test splits with strict cross-split leakage prevention.

        Returns:
            Tuple of (train_records, val_records, test_records)
        """
        # 1. Load train split
        train_records = self.load_split("train")
        train_filenames = {r.filename for r in train_records}

        # 2. Load validation split, explicitly excluding any filename already present in train
        val_records = self.load_split("val", exclude_filenames=train_filenames)
        val_filenames = {r.filename for r in val_records}

        # 3. Load test split, explicitly excluding any filename present in train or val
        combined_seen = train_filenames | val_filenames
        test_records = self.load_split("test", exclude_filenames=combined_seen)
        test_filenames = {r.filename for r in test_records}

        # Strict assertion of zero cross-split overlap
        assert not (train_filenames & val_filenames), "Critical: Train and Val have overlapping samples!"
        assert not (train_filenames & test_filenames), "Critical: Train and Test have overlapping samples!"
        assert not (val_filenames & test_filenames), "Critical: Val and Test have overlapping samples!"

        return train_records, val_records, test_records


def get_split_statistics(records: List[DocumentRecord]) -> Dict[str, Any]:
    """Calculate sample counts and class balance for a split."""
    total = len(records)
    if total == 0:
        return {"total": 0, "genuine": 0, "forged": 0, "genuine_pct": 0.0, "forged_pct": 0.0}

    genuine = sum(1 for r in records if r.label == 0)
    forged = sum(1 for r in records if r.label == 1)

    return {
        "total": total,
        "genuine": genuine,
        "forged": forged,
        "genuine_pct": round((genuine / total) * 100, 2),
        "forged_pct": round((forged / total) * 100, 2),
    }

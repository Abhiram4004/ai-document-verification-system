"""Precompute and cache features across train, val, and test splits."""

import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CACHE_DIR, DATASET_ROOT
from src.data_loader import DatasetLoader
from src.feature_extraction import FeatureExtractor
from src.ocr import OCRPipeline, ensure_tesseract_available
from train import extract_features_for_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


def main():
    print("=" * 65)
    print("      PRECOMPUTING OCR AND VISUAL FEATURES FOR DATASET")
    print("=" * 65)

    # 1. Ensure Tesseract
    tess_cmd = ensure_tesseract_available()
    logger.info("Using Tesseract OCR at: %s", tess_cmd)

    # 2. Load dataset splits with zero-leakage protection
    loader = DatasetLoader(DATASET_ROOT)
    train_records, val_records, test_records = loader.load_all_splits()
    logger.info(
        "Splits loaded: Train=%d, Val=%d, Test=%d (Total=%d)",
        len(train_records),
        len(val_records),
        len(test_records),
        len(train_records) + len(val_records) + len(test_records),
    )

    # 3. Setup feature extractor with persistent OCR cache
    ocr_pipeline = OCRPipeline(tesseract_cmd=tess_cmd, use_cache=True)
    extractor = FeatureExtractor(ocr_pipeline=ocr_pipeline)

    start_time = time.time()

    # Precompute Train
    x_train, y_train = extract_features_for_split(train_records, "train", extractor, CACHE_DIR)
    logger.info("Train features ready: %s labels: %s", x_train.shape, y_train.shape)

    # Precompute Validation
    x_val, y_val = extract_features_for_split(val_records, "val", extractor, CACHE_DIR)
    logger.info("Validation features ready: %s labels: %s", x_val.shape, y_val.shape)

    # Precompute Test
    x_test, y_test = extract_features_for_split(test_records, "test", extractor, CACHE_DIR)
    logger.info("Test features ready: %s labels: %s", x_test.shape, y_test.shape)

    duration = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Feature precomputation complete in {duration:.1f}s ({duration/60:.1f} minutes)!")
    print(f"All features cached in: {CACHE_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()

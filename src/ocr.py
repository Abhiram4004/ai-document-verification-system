"""OCR pipeline for AI Document Verification System using Tesseract OCR."""

import json
import logging
import os
import re
import string
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.config import CACHE_DIR, TESSERACT_CMD, get_tesseract_cmd
from src.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

try:
    import pytesseract
except ImportError:
    pytesseract = None


class TesseractNotFoundError(RuntimeError):
    """Raised when Tesseract OCR binary is missing or cannot be executed."""
    pass


def get_configured_tesseract_cmd() -> Optional[str]:
    """Retrieve verified Tesseract command path."""
    cmd = get_tesseract_cmd() or TESSERACT_CMD
    if cmd and pytesseract is not None:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return cmd


def is_tesseract_available(tesseract_cmd: Optional[str] = None) -> bool:
    """Check whether Tesseract is installed and runnable."""
    if pytesseract is None:
        return False
    cmd = tesseract_cmd or get_configured_tesseract_cmd()
    if not cmd or not Path(cmd).is_file():
        return False
    try:
        pytesseract.pytesseract.tesseract_cmd = cmd
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def ensure_tesseract_available(tesseract_cmd: Optional[str] = None) -> str:
    """Validate Tesseract availability or raise a detailed actionable exception."""
    cmd = tesseract_cmd or get_configured_tesseract_cmd()
    if not cmd or not is_tesseract_available(cmd):
        raise TesseractNotFoundError(
            "Tesseract OCR executable not found or not functional.\n"
            "Tesseract is a mandatory requirement for the training and inference pipeline.\n"
            "To install on Windows:\n"
            "  1. Run: winget install UB-Mannheim.TesseractOCR\n"
            "  2. Or download installer from: https://github.com/UB-Mannheim/tesseract/wiki\n"
            "  3. Configure TESSERACT_CMD in your .env file, e.g.:\n"
            r"     TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
    return cmd



class OCRPipeline:
    """Tesseract OCR processing pipeline with feature extraction and persistent caching."""

    def __init__(
        self,
        tesseract_cmd: Optional[str] = None,
        cache_file: Optional[Path] = None,
        use_cache: bool = True,
    ):
        self.tesseract_cmd = tesseract_cmd or get_configured_tesseract_cmd()
        if self.tesseract_cmd and pytesseract is not None:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

        self.cache_file = cache_file or (CACHE_DIR / "ocr_cache.json")
        self.use_cache = use_cache
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.preprocessor = ImagePreprocessor()
        self._load_cache()

    def _load_cache(self) -> None:
        """Load OCR cache from disk if available."""
        if self.use_cache and self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info("Loaded %d cached OCR entries from %s", len(self.cache), self.cache_file.name)
            except Exception as e:
                logger.warning("Failed to load OCR cache from %s: %e", self.cache_file, e)
                self.cache = {}

    def save_cache(self) -> None:
        """Persist in-memory OCR cache to disk."""
        if not self.use_cache:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
            logger.info("Saved %d OCR cache entries to %s", len(self.cache), self.cache_file.name)
        except Exception as e:
            logger.warning("Failed to save OCR cache to %s: %s", self.cache_file, e)

    @staticmethod
    def compute_text_statistics(raw_text: str, word_confidences: Optional[list] = None) -> Dict[str, float]:
        """Derive deterministic numerical features from extracted OCR text and confidences."""
        text = raw_text.strip()
        char_count = len(text)
        words = text.split()
        word_count = len(words)
        lines = [line for line in text.splitlines() if line.strip()]
        line_count = len(lines)

        digit_count = sum(1 for c in text if c.isdigit())
        alpha_count = sum(1 for c in text if c.isalpha())
        whitespace_count = sum(1 for c in text if c.isspace())
        punct_count = sum(1 for c in text if c in string.punctuation)
        uppercase_count = sum(1 for c in text if c.isupper())

        digit_ratio = digit_count / max(char_count, 1)
        alpha_ratio = alpha_count / max(char_count, 1)
        punct_ratio = punct_count / max(char_count, 1)
        avg_word_length = char_count / max(word_count, 1)
        uppercase_ratio = uppercase_count / max(alpha_count, 1)

        # Confidence statistics
        if word_confidences and len(word_confidences) > 0:
            valid_confs = [c for c in word_confidences if c >= 0]
            mean_conf = float(np.mean(valid_confs)) if valid_confs else 0.0
            low_conf_words = sum(1 for c in valid_confs if c < 50.0)
            low_conf_ratio = low_conf_words / max(len(valid_confs), 1)
        else:
            mean_conf = 0.0
            low_conf_ratio = 0.0

        return {
            "ocr_char_count": float(char_count),
            "ocr_word_count": float(word_count),
            "ocr_line_count": float(line_count),
            "ocr_digit_count": float(digit_count),
            "ocr_alpha_count": float(alpha_count),
            "ocr_punct_count": float(punct_count),
            "ocr_whitespace_count": float(whitespace_count),
            "ocr_digit_ratio": float(digit_ratio),
            "ocr_alpha_ratio": float(alpha_ratio),
            "ocr_punct_ratio": float(punct_ratio),
            "ocr_avg_word_length": float(avg_word_length),
            "ocr_uppercase_ratio": float(uppercase_ratio),
            "ocr_mean_confidence": float(mean_conf),
            "ocr_low_conf_ratio": float(low_conf_ratio),
        }

    def process_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        cache_key: Optional[str] = None,
        mock_text: Optional[str] = None,
    ) -> Tuple[str, Dict[str, float]]:
        """Run OCR pipeline on document image, extract clean text and numerical features.

        Args:
            image_input: Raw image input (path, array, PIL, or bytes)
            cache_key: Identifier (e.g. filename) for caching
            mock_text: Mock text string allowed strictly in automated unit tests

        Returns:
            Tuple of (cleaned_text, text_statistics_dict)
        """
        # 1. Check cache
        if self.use_cache and cache_key and cache_key in self.cache:
            entry = self.cache[cache_key]
            return entry["text"], entry["stats"]

        # 2. Allow mocking strictly for unit tests
        if mock_text is not None:
            stats = self.compute_text_statistics(mock_text)
            return mock_text, stats

        # 3. Ensure real Tesseract is present for production / training / inference
        ensure_tesseract_available(self.tesseract_cmd)

        # 4. Preprocess image for OCR
        preprocessed = self.preprocessor.preprocess_for_ocr(image_input)
        pil_img = Image.fromarray(preprocessed)

        # 5. Execute Tesseract OCR
        try:
            # Extract detailed data including word confidences
            ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
            raw_text = pytesseract.image_to_string(pil_img)

            # Filter valid word confidences
            word_confidences = []
            if "conf" in ocr_data:
                for c in ocr_data["conf"]:
                    try:
                        conf_val = float(c)
                        if conf_val >= 0:
                            word_confidences.append(conf_val)
                    except (ValueError, TypeError):
                        pass

            cleaned_text = re.sub(r"[ \t]+", " ", raw_text).strip()
            stats = self.compute_text_statistics(cleaned_text, word_confidences)

        except Exception as e:
            logger.error("Error during Tesseract OCR extraction: %s", e)
            raise RuntimeError(f"Tesseract OCR failed processing image: {e}") from e

        # 6. Save to cache
        if self.use_cache and cache_key:
            self.cache[cache_key] = {"text": cleaned_text, "stats": stats}

        return cleaned_text, stats

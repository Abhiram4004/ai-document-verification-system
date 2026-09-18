"""Unit tests for OCR pipeline and text statistics extraction."""

import pytest
from src.ocr import OCRPipeline, TesseractNotFoundError, ensure_tesseract_available


def test_text_statistics_calculation():
    """Test deterministic OCR text statistics computation."""
    text = "INVOICE #1024\nDATE 2024-01-15\nTOTAL: $50.00"
    confs = [95.0, 90.0, 88.0, 92.0, 45.0, 89.0]  # One low confidence word (45.0)

    stats = OCRPipeline.compute_text_statistics(text, confs)

    assert stats["ocr_char_count"] == len(text.strip())
    assert stats["ocr_word_count"] == 6
    assert stats["ocr_line_count"] == 3
    assert stats["ocr_digit_count"] == 16  # 1,0,2,4,2,0,2,4,0,1,1,5,5,0,0,0
    assert stats["ocr_digit_ratio"] > 0
    assert stats["ocr_mean_confidence"] > 80.0
    assert stats["ocr_low_conf_ratio"] == pytest.approx(1 / 6, 0.01)


def test_ocr_pipeline_mock(sample_image):
    """Test OCR processing with mock text strictly for isolated unit testing."""
    ocr = OCRPipeline(use_cache=False)
    mock_text = "RECEIPT 123 TOTAL 45.00"

    text, stats = ocr.process_image(sample_image, mock_text=mock_text)

    assert text == mock_text
    assert stats["ocr_word_count"] == 4
    assert stats["ocr_digit_count"] == 7  # 1,2,3,4,5,0,0


def test_tesseract_missing_error(sample_image):
    """Verify that absent Tesseract binary raises clear TesseractNotFoundError when not mocked."""
    # When fake binary path is configured
    ocr = OCRPipeline(tesseract_cmd="C:/nonexistent/tesseract.exe", use_cache=False)
    with pytest.raises((TesseractNotFoundError, RuntimeError)):
        ocr.process_image(image_input=sample_image)


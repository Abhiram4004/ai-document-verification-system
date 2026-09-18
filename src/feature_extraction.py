"""Deterministic feature extraction combining classical computer vision and OCR metrics."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.ocr import OCRPipeline
from src.preprocessing import ImagePreprocessor

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None

# Canonical ordered list of feature names (guarantees training/inference feature consistency)
FEATURE_NAMES: List[str] = [
    # Geometric
    "geom_aspect_ratio",
    "geom_log_area",
    "geom_norm_width",
    "geom_norm_height",
    # Intensity & Contrast
    "gray_mean",
    "gray_std",
    "gray_median",
    "gray_p10",
    "gray_p90",
    "gray_dynamic_range",
    "gray_skewness",
    # Sharpness & Edge Consistency
    "laplacian_var",
    "patch_sharpness_std",
    "patch_sharpness_ratio",
    "canny_edge_density",
    "sobel_h_mean",
    "sobel_v_mean",
    "sobel_grad_energy",
    # Color & Channel Consistency
    "color_red_mean",
    "color_green_mean",
    "color_blue_mean",
    "color_red_std",
    "color_green_std",
    "color_blue_std",
    "color_imbalance",
    "color_sat_mean",
    "color_sat_std",
    # OCR & Text Structure
    "ocr_char_count",
    "ocr_word_count",
    "ocr_line_count",
    "ocr_digit_count",
    "ocr_digit_ratio",
    "ocr_avg_word_length",
    "ocr_uppercase_ratio",
    "ocr_mean_confidence",
    "ocr_low_conf_ratio",
    "ocr_char_density",
]


class FeatureExtractor:
    """Extracts unified classical CV and OCR features from document images."""

    def __init__(self, ocr_pipeline: Optional[OCRPipeline] = None):
        self.preprocessor = ImagePreprocessor()
        self.ocr_pipeline = ocr_pipeline or OCRPipeline()

    @staticmethod
    def _compute_image_features(rgb_image: np.ndarray, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract deterministic visual, texture, and edge features from RGB and Grayscale arrays."""
        h, w = gray_image.shape[:2]
        area = float(h * w)

        # 1. Geometry
        aspect_ratio = float(w) / max(float(h), 1.0)
        log_area = float(np.log(area + 1.0))
        norm_w = float(w) / 1000.0
        norm_h = float(h) / 1000.0

        # 2. Intensity & Contrast
        gray_f = gray_image.astype(np.float32)
        gray_mean = float(np.mean(gray_f))
        gray_std = float(np.std(gray_f))
        gray_median = float(np.median(gray_f))
        p10, p90 = np.percentile(gray_f, [10, 90])
        dynamic_range = float(p90 - p10)

        # Skewness
        if gray_std > 1e-4:
            skewness = float(np.mean(((gray_f - gray_mean) / gray_std) ** 3))
        else:
            skewness = 0.0

        # 3. Sharpness & Edge Consistency
        if cv2 is not None:
            laplacian = cv2.Laplacian(gray_image, cv2.CV_32F)
            laplacian_var = float(laplacian.var())

            # Patch-level sharpness consistency (check for spliced/uneven focus)
            patch_h, patch_w = max(1, h // 4), max(1, w // 4)
            patch_vars = []
            for r in range(4):
                for c in range(4):
                    patch = laplacian[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
                    if patch.size > 0:
                        patch_vars.append(float(patch.var()))
            patch_std = float(np.std(patch_vars)) if patch_vars else 0.0
            patch_ratio = float(max(patch_vars)) / max(min(patch_vars), 1e-4) if patch_vars else 1.0

            # Canny Edge Density
            edges = cv2.Canny(gray_image, 100, 200)
            edge_density = float(np.count_nonzero(edges)) / max(area, 1.0)

            # Sobel Gradients
            sobel_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
            sobel_h_mean = float(np.mean(np.abs(sobel_x)))
            sobel_v_mean = float(np.mean(np.abs(sobel_y)))
            grad_energy = float(np.mean(np.sqrt(sobel_x**2 + sobel_y**2)))
        else:
            # Pure numpy fallback
            diff_x = np.diff(gray_f, axis=1)
            diff_y = np.diff(gray_f, axis=0)
            laplacian_var = float(np.var(diff_x) + np.var(diff_y))
            patch_std = 0.0
            patch_ratio = 1.0
            edge_density = float(np.mean(np.abs(diff_x) > 20))
            sobel_h_mean = float(np.mean(np.abs(diff_x)))
            sobel_v_mean = float(np.mean(np.abs(diff_y)))
            grad_energy = float(np.sqrt(sobel_h_mean**2 + sobel_v_mean**2))

        # 4. Color & Channel Consistency
        r_chan = rgb_image[:, :, 0].astype(np.float32)
        g_chan = rgb_image[:, :, 1].astype(np.float32)
        b_chan = rgb_image[:, :, 2].astype(np.float32)

        r_mean, g_mean, b_mean = float(np.mean(r_chan)), float(np.mean(g_chan)), float(np.mean(b_chan))
        r_std, g_std, b_std = float(np.std(r_chan)), float(np.std(g_chan)), float(np.std(b_chan))
        color_imbalance = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(b_mean - r_mean))

        if cv2 is not None:
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            sat_chan = hsv[:, :, 1].astype(np.float32)
            sat_mean = float(np.mean(sat_chan))
            sat_std = float(np.std(sat_chan))
        else:
            max_c = np.maximum(np.maximum(r_chan, g_chan), b_chan)
            min_c = np.minimum(np.minimum(r_chan, g_chan), b_chan)
            sat_chan = np.where(max_c == 0, 0, (max_c - min_c) / (max_c + 1e-5))
            sat_mean = float(np.mean(sat_chan) * 255.0)
            sat_std = float(np.std(sat_chan) * 255.0)

        return {
            "geom_aspect_ratio": aspect_ratio,
            "geom_log_area": log_area,
            "geom_norm_width": norm_w,
            "geom_norm_height": norm_h,
            "gray_mean": gray_mean,
            "gray_std": gray_std,
            "gray_median": gray_median,
            "gray_p10": float(p10),
            "gray_p90": float(p90),
            "gray_dynamic_range": dynamic_range,
            "gray_skewness": skewness,
            "laplacian_var": laplacian_var,
            "patch_sharpness_std": patch_std,
            "patch_sharpness_ratio": patch_ratio,
            "canny_edge_density": edge_density,
            "sobel_h_mean": sobel_h_mean,
            "sobel_v_mean": sobel_v_mean,
            "sobel_grad_energy": grad_energy,
            "color_red_mean": r_mean,
            "color_green_mean": g_mean,
            "color_blue_mean": b_mean,
            "color_red_std": r_std,
            "color_green_std": g_std,
            "color_blue_std": b_std,
            "color_imbalance": color_imbalance,
            "color_sat_mean": sat_mean,
            "color_sat_std": sat_std,
        }

    def extract_features(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        cache_key: Optional[str] = None,
        mock_ocr_text: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, float], str]:
        """Extract unified feature vector and metadata.

        Args:
            image_input: Raw document image
            cache_key: Optional cache identifier (filename)
            mock_ocr_text: Mock text strictly for testing without Tesseract

        Returns:
            Tuple of (feature_vector_array, feature_dict, extracted_ocr_text)
        """
        # 1. Preprocess image
        rgb_img, gray_img = self.preprocessor.preprocess_for_features(image_input)

        # 2. Extract visual features
        cv_features = self._compute_image_features(rgb_img, gray_img)

        # 3. Extract OCR text and text statistics
        ocr_text, ocr_stats = self.ocr_pipeline.process_image(
            rgb_img, cache_key=cache_key, mock_text=mock_ocr_text
        )

        # Compute character density relative to image size
        h, w = gray_img.shape[:2]
        area_norm = (h * w) / 10000.0
        char_density = ocr_stats["ocr_char_count"] / max(area_norm, 1e-4)

        # 4. Merge features
        merged: Dict[str, float] = {**cv_features}
        merged["ocr_char_count"] = ocr_stats["ocr_char_count"]
        merged["ocr_word_count"] = ocr_stats["ocr_word_count"]
        merged["ocr_line_count"] = ocr_stats["ocr_line_count"]
        merged["ocr_digit_count"] = ocr_stats["ocr_digit_count"]
        merged["ocr_digit_ratio"] = ocr_stats["ocr_digit_ratio"]
        merged["ocr_avg_word_length"] = ocr_stats["ocr_avg_word_length"]
        merged["ocr_uppercase_ratio"] = ocr_stats["ocr_uppercase_ratio"]
        merged["ocr_mean_confidence"] = ocr_stats["ocr_mean_confidence"]
        merged["ocr_low_conf_ratio"] = ocr_stats["ocr_low_conf_ratio"]
        merged["ocr_char_density"] = float(char_density)

        # 5. Assemble ordered feature vector
        vector = np.array([merged[name] for name in FEATURE_NAMES], dtype=np.float32)

        # Safeguard against any unexpected NaN or Inf
        vector = np.nan_to_num(vector, nan=0.0, posinf=1e6, neginf=-1e6)

        return vector, merged, ocr_text

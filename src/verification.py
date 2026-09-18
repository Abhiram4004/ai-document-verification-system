"""Inference engine for document verification."""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from src.config import MODELS_DIR
from src.feature_extraction import FEATURE_NAMES, FeatureExtractor
from src.model import load_pipeline
from src.ocr import OCRPipeline

logger = logging.getLogger(__name__)


class DocumentVerifier:
    """End-to-end inference service for verifying uploaded documents."""

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        ocr_pipeline: Optional[OCRPipeline] = None,
    ):
        self.models_dir = Path(models_dir or MODELS_DIR)
        self.model, self.scaler, self.feature_names, self.metadata = load_pipeline(self.models_dir)
        self.feature_extractor = FeatureExtractor(ocr_pipeline=ocr_pipeline)

        # Calibrated decision threshold (default 0.25 if not specified in metadata)
        self.decision_threshold = float(self.metadata.get("optimal_threshold", 0.25))

        # Optional Visual Deep Learning Model (ResNet-18)
        self.visual_model = None
        self.visual_transform = None
        self._init_visual_model()

    def _init_visual_model(self):
        """Lazily initialize PyTorch visual transfer learning model if checkpoint exists."""
        visual_file = os.getenv("VISUAL_MODEL_FILE", "visual_verifier_resnet18.pt")
        visual_path = self.models_dir / visual_file
        if visual_path.is_file():
            try:
                import torch
                from src.visual_model import DocumentForgeryResNet, get_visual_transforms

                ckpt = torch.load(visual_path, map_location="cpu", weights_only=False)
                image_size = ckpt.get("image_size", 384)
                _, val_transform = get_visual_transforms(image_size=image_size)

                model = DocumentForgeryResNet(pretrained=False)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()

                self.visual_model = model
                self.visual_transform = val_transform
                self.visual_threshold = float(ckpt.get("optimal_threshold", 0.48))
                logger.info("Visual ResNet-18 model loaded successfully into DocumentVerifier.")
            except Exception as e:
                logger.warning("Could not load visual ResNet-18 model: %s", e)

    def verify(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        cache_key: Optional[str] = None,
        mock_ocr_text: Optional[str] = None,
        generate_heatmap: bool = False,
    ) -> Dict[str, Any]:
        """Run complete verification pipeline on a document image.

        Args:
            image_input: File path, bytes, PIL Image, or numpy array
            cache_key: Optional cache identifier
            mock_ocr_text: Optional mock text strictly for unit testing
            generate_heatmap: Whether to compute Grad-CAM attention heatmap

        Returns:
            Structured prediction dictionary
        """
        start_time = time.perf_counter()

        # 1. Feature extraction (visual + OCR)
        feat_vector, feat_dict, ocr_text = self.feature_extractor.extract_features(
            image_input, cache_key=cache_key, mock_ocr_text=mock_ocr_text
        )

        # 2. Reshape and scale features
        feat_matrix = feat_vector.reshape(1, -1)
        feat_scaled = self.scaler.transform(feat_matrix)

        # 3. Primary tabular model inference
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feat_scaled)[0]
            forged_prob = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            df = float(self.model.decision_function(feat_scaled)[0])
            forged_prob = 1.0 / (1.0 + np.exp(-df))
        else:
            forged_prob = float(self.model.predict(feat_scaled)[0])

        # Apply calibrated decision threshold
        is_forged = forged_prob >= self.decision_threshold
        label = 1 if is_forged else 0
        confidence = forged_prob if is_forged else (1.0 - forged_prob)

        # 4. Visual Deep Learning Model Inference (if available)
        visual_prob = None
        heatmap_data = None
        if self.visual_model is not None:
            try:
                import torch
                from src.visual_model import compute_gradcam

                # Load PIL image for visual model
                if isinstance(image_input, (str, Path)):
                    pil_img = Image.open(image_input).convert("RGB")
                elif isinstance(image_input, bytes):
                    import io
                    pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
                elif isinstance(image_input, Image.Image):
                    pil_img = image_input.convert("RGB")
                elif isinstance(image_input, np.ndarray):
                    pil_img = Image.fromarray(image_input).convert("RGB")
                else:
                    pil_img = None

                if pil_img is not None and self.visual_transform is not None:
                    img_tensor = self.visual_transform(pil_img).unsqueeze(0)
                    with torch.no_grad():
                        v_logits = self.visual_model(img_tensor)
                        v_probs = torch.softmax(v_logits, dim=1)[0]
                        visual_prob = round(float(v_probs[1]), 4)

                    if generate_heatmap:
                        cam = compute_gradcam(self.visual_model, img_tensor, target_class=1)
                        heatmap_data = cam.tolist()

            except Exception as e:
                logger.warning("Visual model evaluation error: %s", e)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        prediction_text = "Forged" if is_forged else "Genuine"

        result = {
            "success": True,
            "prediction": prediction_text,
            "label": label,
            "is_forged": is_forged,
            "confidence": round(confidence, 4),
            "forged_probability": round(forged_prob, 4),
            "decision_threshold": self.decision_threshold,
            "ocr_text": ocr_text,
            "processing_time_ms": duration_ms,
            "processing": {
                "ocr_characters": int(feat_dict.get("ocr_char_count", 0)),
                "word_count": int(feat_dict.get("ocr_word_count", 0)),
                "line_count": int(feat_dict.get("ocr_line_count", 0)),
                "digit_count": int(feat_dict.get("ocr_digit_count", 0)),
                "mean_ocr_confidence": round(feat_dict.get("ocr_mean_confidence", 0.0), 2),
            },
            "features": {k: round(v, 4) for k, v in feat_dict.items()},
            "disclaimer": "Model probability represents statistical estimation and is not a legal or certified guarantee of authenticity.",
        }

        if visual_prob is not None:
            result["visual_model"] = {
                "architecture": "ResNet-18 (Transfer Learning)",
                "forged_probability": visual_prob,
                "threshold": getattr(self, "visual_threshold", 0.48),
                "is_forged": visual_prob >= getattr(self, "visual_threshold", 0.48),
            }

        if heatmap_data is not None:
            result["gradcam_attention_map"] = {
                "dimensions": [len(heatmap_data), len(heatmap_data[0]) if heatmap_data else 0],
                "disclaimer": "Grad-CAM attention heatmap illustrates model visual attention patterns and does not represent verified proof or bounding boxes of document forgery.",
            }

        return result

    def generate_gradcam_overlay(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        alpha: float = 0.45,
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray], Optional[float]]:
        """Generate a blended Grad-CAM attention heatmap overlay for the document.

        Args:
            image_input: File path, bytes, PIL Image, or numpy array
            alpha: Heatmap blend opacity (0.0 to 1.0)

        Returns:
            Tuple of (blended_pil_image, raw_cam_numpy_array, forged_probability)
        """
        if self.visual_model is None or self.visual_transform is None:
            return None, None, None

        try:
            import io
            import torch
            from src.visual_model import compute_gradcam, overlay_gradcam

            if isinstance(image_input, (str, Path)):
                pil_img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, bytes):
                pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
            elif isinstance(image_input, Image.Image):
                pil_img = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray):
                pil_img = Image.fromarray(image_input).convert("RGB")
            else:
                return None, None, None

            img_tensor = self.visual_transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                v_logits = self.visual_model(img_tensor)
                v_probs = torch.softmax(v_logits, dim=1)[0]
                visual_prob = float(v_probs[1])

            cam = compute_gradcam(self.visual_model, img_tensor, target_class=1)
            overlay = overlay_gradcam(pil_img, cam, alpha=alpha)
            return overlay, cam, visual_prob
        except Exception as e:
            logger.warning("Error generating Grad-CAM overlay: %s", e)
            return None, None, None


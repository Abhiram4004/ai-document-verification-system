import base64
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FLASK_DEBUG, FLASK_HOST, FLASK_PORT, MODELS_DIR, OUTPUTS_DIR
from src.model import get_model_interpretation
from src.ocr import is_tesseract_available
from src.verification import DocumentVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("api")

try:
    from flask import Flask, jsonify, request, send_file
except ImportError:
    Flask = None


def create_app(verifier: Optional[DocumentVerifier] = None) -> Any:
    """Application factory for Flask REST API."""
    if Flask is None:
        raise ImportError("Flask is not installed. Please install flask to run api.py.")

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

    # Load verifier lazily if not provided
    app_verifier = verifier
    if app_verifier is None:
        try:
            app_verifier = DocumentVerifier(models_dir=MODELS_DIR)
            logger.info("DocumentVerifier successfully loaded into Flask API.")
        except Exception as e:
            logger.warning("Could not pre-load DocumentVerifier: %s (will try on-demand)", e)

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        tess_ok = is_tesseract_available()
        visual_loaded = app_verifier is not None and getattr(app_verifier, "visual_model", None) is not None
        return jsonify({
            "status": "healthy" if (app_verifier is not None and tess_ok) else "degraded",
            "model_loaded": app_verifier is not None,
            "model_name": app_verifier.metadata.get("model_name", "Unknown") if app_verifier else None,
            "visual_model_loaded": visual_loaded,
            "visual_model_architecture": "ResNet-18 (Transfer Learning)" if visual_loaded else None,
            "tesseract_available": tess_ok,
            "endpoints": ["/health", "/model-info", "/predict", "/gradcam"],
        }), 200

    @app.route("/model-info", methods=["GET"])
    def model_info():
        """Model specifications, feature list, and evaluation metrics."""
        if app_verifier is None:
            return jsonify({
                "success": False,
                "error": "Model not loaded. Ensure python train.py has been run.",
            }), 503

        meta = app_verifier.metadata
        interpretation = get_model_interpretation(app_verifier.model, app_verifier.feature_names)

        # Load improved evaluation comparison if available
        improved_eval = {}
        eval_path = OUTPUTS_DIR / "final_test_evaluation.json"
        if eval_path.is_file():
            try:
                with open(eval_path, "r", encoding="utf-8") as f:
                    improved_eval = json.load(f)
            except Exception as e:
                logger.warning("Could not read final_test_evaluation.json: %s", e)

        # Load visual model metadata if available
        visual_meta = {}
        vmeta_path = MODELS_DIR / "visual_metadata.json"
        if vmeta_path.is_file():
            try:
                with open(vmeta_path, "r", encoding="utf-8") as f:
                    visual_meta = json.load(f)
            except Exception as e:
                logger.warning("Could not read visual_metadata.json: %s", e)

        return jsonify({
            "success": True,
            "model_name": meta.get("model_name"),
            "training_timestamp": meta.get("timestamp"),
            "num_features": meta.get("num_features", len(app_verifier.feature_names)),
            "feature_names": app_verifier.feature_names,
            "test_metrics": meta.get("metrics", {}).get("test_metrics", {}),
            "baseline_validation_metrics": meta.get("metrics", {}).get("validation_comparison", {}),
            "baseline_test_metrics": meta.get("metrics", {}).get("test_metrics", {}),
            "improved_test_evaluation": improved_eval,
            "visual_model_metadata": visual_meta,
            "interpretation": interpretation,
            "disclaimer": meta.get("probability_notice"),
        }), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        """Predict document authenticity from uploaded image."""
        # 1. Validate request input first
        if "image" not in request.files and "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "Missing image file. Upload file with key 'image' or 'file'.",
            }), 400

        file = request.files.get("image") or request.files.get("file")
        if file.filename == "":
            return jsonify({
                "success": False,
                "error": "No file selected for upload.",
            }), 400

        allowed_extensions = {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        ext = Path(file.filename).suffix.lower()
        if ext not in allowed_extensions:
            return jsonify({
                "success": False,
                "error": f"Unsupported file extension '{ext}'. Supported: {', '.join(allowed_extensions)}",
            }), 400

        # 2. Check verifier availability
        nonlocal app_verifier
        if app_verifier is None:
            try:
                app_verifier = DocumentVerifier(models_dir=MODELS_DIR)
            except Exception as e:
                return jsonify({
                    "success": False,
                    "error": f"Verification pipeline unavailable: {str(e)}",
                    "hint": "Ensure models have been trained with 'python train.py'",
                }), 503

        try:
            image_bytes = file.read()
            include_gradcam = request.args.get("gradcam", "").lower() in ("true", "1", "yes")
            result = app_verifier.verify(image_bytes, generate_heatmap=include_gradcam)

            if include_gradcam and getattr(app_verifier, "visual_model", None) is not None:
                overlay, _, _ = app_verifier.generate_gradcam_overlay(image_bytes)
                if overlay is not None:
                    buf = io.BytesIO()
                    overlay.save(buf, format="PNG")
                    result["gradcam_overlay_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
                    result["gradcam_disclaimer"] = (
                        "Grad-CAM attention heatmap illustrates model visual attention patterns "
                        "and does not represent verified proof or bounding boxes of document forgery."
                    )

            return jsonify(result), 200
        except Exception as e:
            logger.error("Prediction error: %s", e)
            return jsonify({
                "success": False,
                "error": f"Internal prediction error: {str(e)}",
            }), 500

    @app.route("/gradcam", methods=["POST"])
    def get_gradcam_image():
        """Generate and stream blended Grad-CAM attention heatmap image."""
        if "image" not in request.files and "file" not in request.files:
            return jsonify({"error": "Missing image file."}), 400

        file = request.files.get("image") or request.files.get("file")
        if file.filename == "":
            return jsonify({"error": "No file selected."}), 400

        nonlocal app_verifier
        if app_verifier is None:
            try:
                app_verifier = DocumentVerifier(models_dir=MODELS_DIR)
            except Exception as e:
                return jsonify({"error": f"Verifier unavailable: {e}"}), 503

        if app_verifier.visual_model is None:
            return jsonify({"error": "Visual deep learning model is not available."}), 503

        try:
            image_bytes = file.read()
            overlay, _, _ = app_verifier.generate_gradcam_overlay(image_bytes)
            if overlay is None:
                return jsonify({"error": "Could not compute Grad-CAM overlay."}), 500

            buf = io.BytesIO()
            overlay.save(buf, format="PNG")
            buf.seek(0)
            return send_file(buf, mimetype="image/png", download_name="gradcam_overlay.png")
        except Exception as e:
            return jsonify({"error": f"Grad-CAM generation error: {e}"}), 500


    return app


if __name__ == "__main__":
    app = create_app()
    print(f"Starting Flask REST API on http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)

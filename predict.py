"""CLI prediction script for AI Document Verification System."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR
from src.verification import DocumentVerifier

logging.basicConfig(level=logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Verify a document or receipt image for potential forgery.")
    parser.add_argument("--image", type=str, required=True, help="Path to document image file (.png, .jpg, etc.)")
    parser.add_argument("--models-dir", type=str, default=str(MODELS_DIR), help="Directory containing model artifacts")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of formatted text")
    parser.add_argument("--gradcam", type=str, default=None, help="Path to save visual model Grad-CAM attention heatmap (.png)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        verifier = DocumentVerifier(models_dir=Path(args.models_dir))
    except Exception as e:
        print(f"Error loading verification pipeline: {e}", file=sys.stderr)
        print("Ensure models are trained first: run 'python train.py'", file=sys.stderr)
        sys.exit(1)

    try:
        result = verifier.verify(image_path, generate_heatmap=bool(args.gradcam))
    except Exception as e:
        print(f"Error during document verification: {e}", file=sys.stderr)
        sys.exit(1)

    # If --gradcam requested, save heatmap overlay
    if args.gradcam and verifier.visual_model is not None:
        try:
            overlay_img, _, _ = verifier.generate_gradcam_overlay(image_path)
            if overlay_img is not None:
                out_cam = Path(args.gradcam)
                out_cam.parent.mkdir(parents=True, exist_ok=True)
                overlay_img.save(out_cam)
                result["gradcam_saved_to"] = str(out_cam)
            else:
                result["gradcam_error"] = "Visual model unavailable for Grad-CAM generation."
        except Exception as e:
            result["gradcam_error"] = str(e)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    # Formatted console output
    is_forged = result["is_forged"]
    status_text = "[!] FORGED / MANIPULATED" if is_forged else "[OK] GENUINE / AUTHENTIC"

    print("\n" + "=" * 65)
    print("           DOCUMENT VERIFICATION RESULT")
    print("=" * 65)
    print(f"File:                   {image_path.name}")
    print(f"Primary Prediction:     {status_text}")
    print(f"Primary Probability:    {result['confidence'] * 100:.2f}%")
    print(f"Forged Probability:     {result['forged_probability'] * 100:.2f}%")
    print(f"Decision Threshold:     {result.get('decision_threshold', 0.50):.2f}")

    if "visual_model" in result:
        vm = result["visual_model"]
        vm_status = "FORGED" if vm["is_forged"] else "GENUINE"
        print(f"Visual Model ({vm['architecture']}):")
        print(f"  - Prediction:         [{vm_status}] (Prob: {vm['forged_probability']*100:.2f}%, Thresh: {vm['threshold']:.2f})")

    if args.gradcam and "gradcam_saved_to" in result:
        print(f"Grad-CAM Heatmap:       Saved to {result['gradcam_saved_to']}")

    print(f"Processing Time:        {result['processing_time_ms']} ms")
    print("-" * 65)
    print("OCR Processing Summary:")
    print(f"  - Extracted Characters: {result['processing']['ocr_characters']}")
    print(f"  - Extracted Words:      {result['processing']['word_count']}")
    print(f"  - Extracted Digits:     {result['processing']['digit_count']}")
    print(f"  - Mean OCR Confidence:  {result['processing']['mean_ocr_confidence']}%")

    # OCR text preview
    ocr_preview = result["ocr_text"].strip()
    if ocr_preview:
        lines = [line.strip() for line in ocr_preview.splitlines() if line.strip()][:3]
        snippet = " | ".join(lines)
        if len(snippet) > 80:
            snippet = snippet[:77] + "..."
        print(f"  - Text Snippet:         \"{snippet}\"")
    else:
        print("  - Text Snippet:         (No readable text found)")

    print("-" * 65)
    print("Notice:")
    print(f"  {result['disclaimer']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

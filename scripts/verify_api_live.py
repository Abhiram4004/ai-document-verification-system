"""Verification script for all Flask API endpoints."""

import io
import json
import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import create_app

app = create_app()
app.config["TESTING"] = True
client = app.test_client()

# 1. GET /health
r_health = client.get("/health")
print("=== GET /health ===")
print("Status:", r_health.status_code)
d_health = r_health.get_json()
print(json.dumps(d_health, indent=2))
assert r_health.status_code == 200
assert d_health["model_loaded"] is True
assert d_health["visual_model_loaded"] is True
assert d_health["tesseract_available"] is True

# 2. GET /model-info
r_info = client.get("/model-info")
print("\n=== GET /model-info ===")
print("Status:", r_info.status_code)
d_info = r_info.get_json()
print("Model name:", d_info.get("model_name"))
print("Num features:", d_info.get("num_features"))
print("Test metrics:", d_info.get("test_metrics"))
assert r_info.status_code == 200
assert d_info["success"] is True

# 3. POST /predict
test_img_path = PROJECT_ROOT / "archive" / "findit2" / "test" / "X00016469619.png"
with open(test_img_path, "rb") as f:
    img_bytes = f.read()

r_predict = client.post("/predict", data={"image": (io.BytesIO(img_bytes), "test.png")}, content_type="multipart/form-data")
print("\n=== POST /predict ===")
print("Status:", r_predict.status_code)
d_predict = r_predict.get_json()
print("Prediction:", d_predict.get("prediction"))
print("Decision Threshold:", d_predict.get("decision_threshold"))
print("Visual model:", d_predict.get("visual_model"))
assert r_predict.status_code == 200
assert d_predict["decision_threshold"] == 0.25
assert d_predict["visual_model"]["threshold"] == 0.48

# 4. POST /predict?gradcam=true
r_gradcam_flag = client.post("/predict?gradcam=true", data={"image": (io.BytesIO(img_bytes), "test.png")}, content_type="multipart/form-data")
print("\n=== POST /predict?gradcam=true ===")
print("Status:", r_gradcam_flag.status_code)
d_gradcam_flag = r_gradcam_flag.get_json()
assert r_gradcam_flag.status_code == 200
assert "gradcam_overlay_base64" in d_gradcam_flag
assert len(d_gradcam_flag["gradcam_overlay_base64"]) > 100
assert "gradcam_disclaimer" in d_gradcam_flag
print("Grad-CAM base64 length:", len(d_gradcam_flag["gradcam_overlay_base64"]))
print("Disclaimer:", d_gradcam_flag["gradcam_disclaimer"])

# 5. POST /gradcam
r_gradcam = client.post("/gradcam", data={"image": (io.BytesIO(img_bytes), "test.png")}, content_type="multipart/form-data")
print("\n=== POST /gradcam ===")
print("Status:", r_gradcam.status_code)
print("Content-Type:", r_gradcam.content_type)
print("Image bytes length:", len(r_gradcam.data))
assert r_gradcam.status_code == 200
assert r_gradcam.content_type == "image/png"
assert len(r_gradcam.data) > 1000

print("\n>>> ALL FLASK API ENDPOINTS FULLY VERIFIED! <<<")

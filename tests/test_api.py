"""Unit tests for Flask REST API endpoints."""

import io
import pytest
from PIL import Image

try:
    from api import create_app
    from flask.testing import FlaskClient
except ImportError:
    create_app = None


@pytest.fixture
def client():
    """Create Flask test client."""
    if create_app is None:
        pytest.skip("Flask not installed")
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test GET /health returns 200 with status fields."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "model_loaded" in data
    assert "tesseract_available" in data


def test_predict_endpoint_missing_file(client):
    """Test POST /predict without file upload returns 400."""
    response = client.post("/predict")
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert "Missing image file" in data["error"]


def test_predict_endpoint_unsupported_file(client):
    """Test POST /predict with invalid file type returns 400."""
    data = {"image": (io.BytesIO(b"fake text content"), "test.txt")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    res_data = response.get_json()
    assert res_data["success"] is False
    assert "Unsupported file extension" in res_data["error"]


def test_model_info_endpoint(client):
    """Test GET /model-info returns model details and validation/test metrics."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "model_name" in data
    assert "num_features" in data
    assert data["num_features"] == 37
    assert "test_metrics" in data


def test_predict_endpoint_with_image(client):
    """Test POST /predict with an actual image returns 200 and valid verification results."""
    # Create small valid test image in memory
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {"image": (buf, "sample_receipt.png")}
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert res["prediction"] in ("Genuine", "Forged")
    assert "forged_probability" in res
    assert "confidence" in res
    assert "processing" in res
    assert "disclaimer" in res


def test_predict_endpoint_with_gradcam_flag(client):
    """Test POST /predict?gradcam=true returns base64 heatmap overlay."""
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {"image": (buf, "sample_receipt.png")}
    response = client.post("/predict?gradcam=true", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    res = response.get_json()
    assert res["success"] is True
    assert "visual_model" in res
    assert "gradcam_overlay_base64" in res
    assert "gradcam_disclaimer" in res


def test_gradcam_endpoint_image_stream(client):
    """Test POST /gradcam streams PNG image overlay."""
    img = Image.new("RGB", (100, 100), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    data = {"image": (buf, "sample_receipt.png")}
    response = client.post("/gradcam", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.content_type == "image/png"
    assert len(response.data) > 0


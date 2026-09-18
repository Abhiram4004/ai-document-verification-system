"""Configuration manager for AI Document Verification System."""

import os
import shutil
from pathlib import Path

# Load environment variables if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Base repository directory
REPO_ROOT = Path(__file__).resolve().parent.parent

# Dataset root directory
_env_dataset = os.getenv("DATASET_ROOT", "archive/findit2")
DATASET_ROOT = (REPO_ROOT / _env_dataset).resolve() if not Path(_env_dataset).is_absolute() else Path(_env_dataset)

# Directories
MODELS_DIR = (REPO_ROOT / os.getenv("MODELS_DIR", "models")).resolve()
OUTPUTS_DIR = (REPO_ROOT / os.getenv("OUTPUTS_DIR", "outputs")).resolve()
CACHE_DIR = (REPO_ROOT / os.getenv("CACHE_DIR", "data/cache")).resolve()

# Ensure required runtime output/cache directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Tesseract OCR binary auto-discovery
def get_tesseract_cmd() -> str | None:
    """Find the Tesseract executable path via config, PATH, or standard Windows paths."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and Path(env_cmd).is_file():
        return str(Path(env_cmd).resolve())

    # Check PATH
    which_cmd = shutil.which("tesseract")
    if which_cmd:
        return which_cmd

    # Check common Windows installation paths
    candidate_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidate_paths:
        if Path(p).is_file():
            return str(Path(p).resolve())

    return None

TESSERACT_CMD = get_tesseract_cmd()

# Server configurations
FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "yes")

# Model artifacts
SAVED_MODEL_FILE = (MODELS_DIR / os.getenv("CLASSICAL_MODEL_FILE", "document_verifier.joblib")).resolve()
CLASSICAL_MODEL_PATH = SAVED_MODEL_FILE
SAVED_PIPELINE_FILE = (MODELS_DIR / "feature_pipeline.joblib").resolve()
MODEL_METADATA_FILE = (MODELS_DIR / "metadata.json").resolve()
VISUAL_MODEL_PATH = (MODELS_DIR / os.getenv("VISUAL_MODEL_FILE", "visual_verifier_resnet18.pt")).resolve()
VISUAL_METADATA_FILE = (MODELS_DIR / "visual_metadata.json").resolve()

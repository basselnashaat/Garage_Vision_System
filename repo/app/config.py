"""
Project configuration file.
All hardcoded values are extracted from the notebook and centralized here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file (skipped on HF Spaces demo)
GRADIO_SPACE = os.getenv("GRADIO_SPACE", "").lower() in ("1", "true", "yes")
if not GRADIO_SPACE:
    ENV_FILE = find_dotenv(filename=".env", usecwd=True)
    if ENV_FILE:
        load_dotenv(ENV_FILE, override=False, interpolate=True)

# ======================
# PROJECT PATHS
# ======================
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
BASE_DIR = str(PROJECT_ROOT)
APP_DIR = Path(__file__).parent.resolve()

# ======================
# MODEL PATHS & HUGGING FACE CONFIGURATION
# ======================
HF_TOKEN = os.getenv("HF_TOKEN", "")
USE_HUGGING_FACE = (
    os.getenv("USE_HUGGING_FACE", "").lower() in ("1", "true", "yes")
    or bool(HF_TOKEN)
    or bool(
        os.getenv("HF_PLATE_DETECTOR_REPO")
        and os.getenv("HF_CHAR_OCR_REPO")
    )
)

# Hugging Face model IDs
HF_PLATE_DETECTOR_REPO = os.getenv("HF_PLATE_DETECTOR_REPO", "")
HF_CHAR_OCR_REPO = os.getenv("HF_CHAR_OCR_REPO", "")
HF_CAR_CLASSIFIER_REPO = os.getenv("HF_CAR_CLASSIFIER_REPO", "")

# Local fallback paths (used if HF token not provided)
MODELS_DIR = PROJECT_ROOT / "models"
PLATE_DETECTOR_MODEL_PATH = str(os.getenv("PLATE_DETECTOR_MODEL_PATH", 
                                           MODELS_DIR / "plate_detector" / "best.pt"))
CHAR_OCR_MODEL_PATH = str(os.getenv("CHAR_OCR_MODEL_PATH", 
                                     MODELS_DIR / "char_ocr" / "best.pt"))
CAR_CLASSIFIER_MODEL_PATH = str(os.getenv("CAR_CLASSIFIER_MODEL_PATH", 
                                          MODELS_DIR / "car_classifier" / "best.pt"))

# Debug: Print resolved paths
import logging
logger = logging.getLogger(__name__)
logger.info(f"Config - PROJECT_ROOT: {PROJECT_ROOT}")
logger.info(f"Config - MODELS_DIR: {MODELS_DIR}")
logger.info(f"Config - PLATE_DETECTOR_MODEL_PATH: {PLATE_DETECTOR_MODEL_PATH}")

# ======================
# MODEL INFERENCE SETTINGS
# ======================
CONF_THRESHOLD = 0.25         # Confidence threshold for YOLO detections
VARIANCE_THRESHOLD = 100.0    # Minimum variance to consider frame usable

# ======================
# IMAGE PREPROCESSING SETTINGS
# ======================
# Enhancer removed: EDSR settings deprecated

# ======================
# SCORING WEIGHTS
# ======================
SEGMENT_SCORE_WEIGHT = 0.65   # Weight for segment score in final calculation
PLATE_SCORE_WEIGHT = 0.35     # Weight for plate score in final calculation

# ======================
# DATABASE CONFIGURATION (Supabase)
# ======================

# Supabase configuration (NEW)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://yoenbdxxdwknizqthkfp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
'''
SUPABASE_URL         = os.getenv("SUPABASE_URL",         "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
'''

if not SUPABASE_SERVICE_KEY and not GRADIO_SPACE:
    raise ValueError("SUPABASE_SERVICE_KEY not found in environment variables")

# ======================
# PLATE SCORING RULES
# ======================
PLATE_SCORING_RULES = {
    "digit_count_weight": 0.3,      # Lower digit count → higher score
    "repeating_weight": 0.2,        # Bonus for repeating patterns
    "sequential_weight": 0.2,       # Bonus for sequential patterns
    "low_number_weight": 0.3,       # Bonus for numbers < 100
}

# ======================
# VIDEO CAPTURE (pipeline VideoCapturer)
# ======================
CAPTURE_OUTPUT_DIR = PROJECT_ROOT / "src" / "capture" / "captured_frames"
CAPTURE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAPTURE_PROCESS_EVERY_N_FRAMES = int(os.getenv("CAPTURE_PROCESS_EVERY_N_FRAMES", "3"))
CAPTURE_FRAME_BUFFER_SIZE = int(os.getenv("CAPTURE_FRAME_BUFFER_SIZE", "20"))
CAPTURE_MOTION_THRESHOLD = int(os.getenv("CAPTURE_MOTION_THRESHOLD", "1500"))
CAPTURE_MOG2_HISTORY = int(os.getenv("CAPTURE_MOG2_HISTORY", "500"))
CAPTURE_MOG2_VAR_THRESHOLD = int(os.getenv("CAPTURE_MOG2_VAR_THRESHOLD", "50"))
CAPTURE_COOLDOWN_SECONDS = float(os.getenv("CAPTURE_COOLDOWN_SECONDS", "2"))
CAPTURE_TRIGGER_MODE = os.getenv("CAPTURE_TRIGGER_MODE", "yolo")  # yolo | motion
CAPTURE_VEHICLE_MODEL = os.getenv("CAPTURE_VEHICLE_MODEL", "yolo11n.pt")
CAPTURE_VEHICLE_CONF = float(os.getenv("CAPTURE_VEHICLE_CONF", "0.40"))
CAPTURE_VEHICLE_MIN_BOX_AREA = int(os.getenv("CAPTURE_VEHICLE_MIN_BOX_AREA", "3000"))
CAPTURE_VEHICLE_CROP_PADDING = int(os.getenv("CAPTURE_VEHICLE_CROP_PADDING", "50"))
CAPTURE_VEHICLE_FILTER_ENABLED = os.getenv("CAPTURE_VEHICLE_FILTER_ENABLED", "true").lower() == "true"
CAPTURE_MAX_VEHICLES_PER_TRIGGER = int(os.getenv("CAPTURE_MAX_VEHICLES_PER_TRIGGER", "1"))
CAPTURE_MIN_SHARPNESS_SCORE = float(os.getenv("CAPTURE_MIN_SHARPNESS_SCORE", "5.0"))
CAPTURE_MIN_BRIGHTNESS = int(os.getenv("CAPTURE_MIN_BRIGHTNESS", "30"))
CAPTURE_MAX_BRIGHTNESS = int(os.getenv("CAPTURE_MAX_BRIGHTNESS", "220"))
CAPTURE_DEDUP_IOU = float(os.getenv("CAPTURE_DEDUP_IOU", "0.45"))
# Safety cap for POST /api/detect video uploads (None = no cap)
CAPTURE_MAX_TRIGGERS = int(os.getenv("CAPTURE_MAX_TRIGGERS", "30")) or None

# ======================
# ENSURE REQUIRED PATHS
# ======================
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────
#  EALPR  —  Video Capture Configuration
# ─────────────────────────────────────────────

# ── Camera Settings ───────────────────────────
# Use 0 for the default webcam (testing)
# Use an RTSP URL for a real IP camera, e.g.:
#   "rtsp://admin:password@192.168.1.100:554/stream1"
# Use a file path to process a local video, e.g.:
#   "1900-151662242.mp4"  or  r"C:\path\to\video.mp4"

# ▼ Switch between sources by commenting / uncommenting:
#CAMERA_URL = r"d:\Uni y3s2\ML\Project_Main_ML\1900-151662242.mp4"  # local video
CAMERA_URL = "http://10.48.140.136:4747/video"   # IP camera (DroidCam)
# CAMERA_URL = 0                                  # default webcam

# Frames per second to READ from the stream
# (does not need to match camera FPS — we sample every N frames)
# OPTIMIZATION: Changed from 3 to 12 to jump over identical redundant frames rapidly
PROCESS_EVERY_N_FRAMES = 12   

# ── Frame Buffer ──────────────────────────────
# How many frames to keep in memory before picking the best one
# More = better chance of catching a sharp frame, but more memory
# OPTIMIZATION: Raised to 45 to give the selector a broader pool to find the sharpest snapshot
FRAME_BUFFER_SIZE = 45

# ── Motion / Trigger Settings ─────────────────
# Minimum number of changed pixels to count as "vehicle present"
MIN_MOTION_AREA     = 500   # Lower = more sensitive to small objects / distance

# Max frame count a vehicle can go without motion before ending the cluster event
MAX_STATIONARY_FRAMES = 15   # Lower = splits slow traffic into separate events

# Quality/Selection rules:
MIN_PLATE_CONFIDENCE = 0.40  # discard frames where plate detection is weak
MAX_PLATE_ANGLE      = 15.0  # degrees max skew allowed (if computed)

# Brightness constraints to filter out garbage flashes
MIN_BRIGHTNESS      = 30    # widened for dark/night scenes in video
MAX_BRIGHTNESS      = 220   # widened for bright scenes in video

# ── Model Settings ────────────────────────────
CONF_THRESHOLD = 0.25
USE_HUGGING_FACE = True

HF_TOKEN = None
HF_PLATE_DETECTOR_REPO = "alyhassankamel/egyptian_license_plate_detector"
HF_CHAR_OCR_REPO       = "alyhassankamel/egyptian-license-plate-ocr"
HF_CAR_CLASSIFIER_REPO = "PatrickBLB/egyptian_cars_recognition_computer_vision_model"

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
PLATE_DETECTOR_MODEL_PATH = str(PROJECT_ROOT / "weights" / "detector_best.pt")
CHAR_OCR_MODEL_PATH       = str(PROJECT_ROOT / "weights" / "ocr_best.pt")

# ── Output / Storage ──────────────────────────
import os

# Directory where captured frames are saved
OUTPUT_DIR = "captured_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save every triggered frame to disk?  True = yes, False = send to API only
SAVE_FRAMES_TO_DISK = True

# Image format for saved frames
SAVE_FORMAT = ".jpg"
JPEG_QUALITY = 95   # 0–100

# ── API Settings ──────────────────────────────
# FastAPI /api/detect endpoint.
# Set to None to skip sending and only save locally.
API_ENDPOINT = "http://localhost:8000/api/detect"   # ← FastAPI server
API_TIMEOUT  = 60     # seconds (LPR pipeline can be slow on CPU)

# Camera UUID registered in the 'cameras' table (Supabase).
# Every detection is tagged with this ID so the dashboard knows
# which physical camera triggered the event.
# Create a camera first:  POST /api/cameras  → copy the returned UUID here.
CAMERA_ID = "70a649c0-4ad7-4a85-bc12-9e7144e4a777"

# ── Display / Debug ───────────────────────────
# Show live preview window while running?  Set False on headless servers
SHOW_PREVIEW = True

# Draw motion mask overlay on the preview?
SHOW_MOTION_MASK = False

# Print verbose console status?
VERBOSE_LOGGING = True
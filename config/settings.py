"""
config/settings.py
Central configuration for the Face Attendance System.
All thresholds, paths, and constants are defined here — no magic numbers in code.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "weights")
FACE_DB_DIR = os.path.join(BASE_DIR, "face_db")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FACE_DB_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ─── Flask ────────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit

# ─── Object Detection (YOLOv8) ────────────────────────────────────────────────
DETECTION_MODEL = "yolov8n.pt"              # General COCO detector
FACE_DETECTION_MODEL = "yolov8n-face.pt"    # Custom face detector weights
DETECTION_CONFIDENCE = 0.45
DETECTION_IOU = 0.50
DETECTION_CLASSES = [0]                     # 0 = person in COCO
DETECTION_DEVICE = None                     # None = auto (cuda if available)

# ─── Object Tracking (ByteTrack / BoTSORT via ultralytics) ─────────────────────
TRACKER_TYPE = "bytetrack"                  # "bytetrack" or "botsort"
TRACKER_CONFIDENCE = 0.40
TRACKER_IOU = 0.50

# ─── Face Recognition (InsightFace / ArcFace) ─────────────────────────────────
RECOGNITION_MODEL = "buffalo_l"             # InsightFace model pack
RECOGNITION_THRESHOLD = 0.60               # Cosine similarity threshold
EMBEDDING_DIM = 512
FACE_DB_PATH = os.path.join(FACE_DB_DIR, "embeddings.pkl")

# ─── Anti-Spoofing ────────────────────────────────────────────────────────────
SPOOF_MODEL_PATH = os.path.join(MODEL_DIR, "minifasnet.onnx")
SPOOF_THRESHOLD = 0.80                     # Probability of being "live"
SPOOF_INPUT_SIZE = (80, 80)
SPOOF_LBP_WEIGHT = 0.6                    # Weight for LBP in heuristic ensemble
SPOOF_FFT_WEIGHT = 0.4                    # Weight for FFT in heuristic ensemble

# ─── Liveness Detection ───────────────────────────────────────────────────────
LIVENESS_BLINK_THRESHOLD = 0.25            # EAR below this = blink
LIVENESS_HEAD_POSE_THRESHOLD = 15.0        # Degrees
LIVENESS_CHALLENGE_FRAMES = 30             # Frames to complete challenge
LIVENESS_SESSION_TIMEOUT = 60              # Seconds before session expires
LIVENESS_MIN_FRAMES_PASS = 5              # Minimum frames before can pass

# ─── Fraud Detection ──────────────────────────────────────────────────────────
FRAUD_DUPLICATE_WINDOW = 300               # Seconds (5 min)
FRAUD_MAX_SPEED_PX_PER_FRAME = 80         # Max realistic face movement
FRAUD_CONSISTENCY_THRESHOLD = 0.90         # Embedding consistency floor
FRAUD_FAILED_ATTEMPTS_WINDOW = 300         # 5 min window for failed attempts
FRAUD_FAILED_ATTEMPTS_LIMIT = 10           # Threshold for excessive failures
FRAUD_UNUSUAL_HOUR_START = 0               # Hour range considered unusual
FRAUD_UNUSUAL_HOUR_END = 5

# ─── Face Quality ─────────────────────────────────────────────────────────────
QUALITY_MIN_RESOLUTION = (60, 60)
QUALITY_MIN_SHARPNESS = 50.0
QUALITY_MIN_BRIGHTNESS = 40.0
QUALITY_MAX_BRIGHTNESS = 220.0
QUALITY_MAX_OCCLUSION = 0.30
QUALITY_POSE_THRESHOLD = 30.0              # Degrees yaw/pitch/roll
QUALITY_ACCEPT_THRESHOLD = 0.55

# ─── Quality Dimension Weights ────────────────────────────────────────────────
QUALITY_WEIGHTS = {
    "resolution": 0.25,
    "sharpness": 0.20,
    "brightness": 0.15,
    "contrast": 0.10,
    "occlusion": 0.15,
    "pose": 0.10,
    "symmetry": 0.05,
}

# ─── Allowed Image Extensions ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

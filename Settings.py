"""
config/settings.py
Central configuration for the Face Attendance System.
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

# ─── Face Recognition ─────────────────────────────────────────────────────────
RECOGNITION_MODEL = "ArcFace"          # backbone
RECOGNITION_THRESHOLD = 0.60           # cosine similarity threshold
EMBEDDING_DIM = 512
FACE_DB_PATH = os.path.join(FACE_DB_DIR, "embeddings.pkl")

# ─── Object Detection ─────────────────────────────────────────────────────────
DETECTION_MODEL = "yolov8n.pt"         # nano YOLOv8 for speed
DETECTION_CONFIDENCE = 0.45
DETECTION_IOU = 0.50
DETECTION_CLASSES = [0]                # 0 = person in COCO

# ─── Object Tracking (ByteTrack / SORT) ───────────────────────────────────────
TRACKER_MAX_AGE = 30                   # frames before track is deleted
TRACKER_MIN_HITS = 3
TRACKER_IOU_THRESHOLD = 0.30

# ─── Anti-Spoofing ────────────────────────────────────────────────────────────
SPOOF_MODEL = "MiniFASNet"
SPOOF_THRESHOLD = 0.80                 # probability of being "live"
SPOOF_INPUT_SIZE = (80, 80)

# ─── Liveness Detection ───────────────────────────────────────────────────────
LIVENESS_BLINK_THRESHOLD = 0.25        # EAR below this = blink
LIVENESS_HEAD_POSE_THRESHOLD = 15.0   # degrees
LIVENESS_CHALLENGE_FRAMES = 30        # frames to complete challenge

# ─── Fraud Detection ──────────────────────────────────────────────────────────
FRAUD_DUPLICATE_WINDOW = 300           # seconds (5 min)
FRAUD_MAX_SPEED_PX_PER_FRAME = 80     # max realistic face movement
FRAUD_CONSISTENCY_THRESHOLD = 0.90

# ─── Face Quality ─────────────────────────────────────────────────────────────
QUALITY_MIN_RESOLUTION = (60, 60)
QUALITY_MIN_SHARPNESS = 50.0
QUALITY_MIN_BRIGHTNESS = 40.0
QUALITY_MAX_BRIGHTNESS = 220.0
QUALITY_MAX_OCCLUSION = 0.30
QUALITY_POSE_THRESHOLD = 30.0         # degrees yaw/pitch/roll

# ─── Allowed Image Extensions ─────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
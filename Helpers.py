"""
utils/helpers.py
Shared utilities: image loading, validation, response builders.
"""

import base64
import io
import logging
import os
import time
from functools import wraps
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from flask import jsonify

from config.settings import ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH

logger = logging.getLogger(__name__)


# ─── Image Utilities ──────────────────────────────────────────────────────────

def decode_image(source) -> Optional[np.ndarray]:
    """
    Accept a Flask FileStorage, raw bytes, base64 string, or numpy array.
    Returns a BGR numpy array (OpenCV convention) or None on failure.
    """
    try:
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            # Strip optional data-URI prefix
            if "," in source:
                source = source.split(",", 1)[1]
            raw = base64.b64decode(source)
            source = io.BytesIO(raw)

        if hasattr(source, "read"):
            raw = source.read()
            source = io.BytesIO(raw)

        pil_img = Image.open(source).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr

    except Exception as exc:
        logger.warning("decode_image failed: %s", exc)
        return None


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def resize_keep_aspect(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def crop_face(img: np.ndarray, bbox, padding: float = 0.15) -> np.ndarray:
    """Crop face region with optional padding."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return img[y1:y2, x1:x2]


# ─── Face Geometry ────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_landmarks) -> float:
    """Compute EAR for blink/liveness detection."""
    # eye_landmarks: 6 points [[x,y], ...]
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C + 1e-6)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ─── Response Builders ────────────────────────────────────────────────────────

def success_response(data: dict, status: int = 200):
    return jsonify({"status": "success", "data": data}), status


def error_response(message: str, status: int = 400):
    return jsonify({"status": "error", "message": message}), status


# ─── Timing Decorator ─────────────────────────────────────────────────────────

def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("%s completed in %.1f ms", fn.__name__, elapsed)
        return result
    return wrapper


# ─── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
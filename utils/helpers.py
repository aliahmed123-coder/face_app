"""
utils/helpers.py
Shared utility functions: image I/O, validation, geometry, response builders, timing.
"""

import base64
import io
import logging
import time
from functools import wraps
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─── Image Utilities ──────────────────────────────────────────────────────────


def decode_image(source) -> Optional[np.ndarray]:
    """
    Decode an image from various sources into a BGR numpy array.

    Accepts:
        - numpy.ndarray (returned as-is)
        - str (base64-encoded, with optional data-URI prefix)
        - file-like object with .read() method (e.g., Flask FileStorage)
        - bytes

    Returns:
        BGR numpy array (OpenCV convention), or None on failure.
    """
    try:
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            # Strip optional data-URI prefix (e.g., "data:image/png;base64,...")
            if "," in source:
                source = source.split(",", 1)[1]
            raw = base64.b64decode(source)
            source = io.BytesIO(raw)

        if isinstance(source, bytes):
            source = io.BytesIO(source)

        if hasattr(source, "read"):
            raw = source.read()
            if isinstance(raw, bytes) and len(raw) == 0:
                return None
            source = io.BytesIO(raw) if isinstance(raw, bytes) else io.BytesIO(raw)

        pil_img = Image.open(source).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return bgr

    except Exception as exc:
        logger.warning("decode_image failed: %s", exc)
        return None


def allowed_file(filename: str) -> bool:
    """Check if a filename has an allowed image extension."""
    from config.settings import ALLOWED_EXTENSIONS

    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_keep_aspect(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Resize image so the longest side is at most `max_side`, preserving aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def crop_face(img: np.ndarray, bbox: List[int], padding: float = 0.15) -> np.ndarray:
    """
    Crop a face region from an image with optional padding.

    Args:
        img: BGR image array.
        bbox: Bounding box [x1, y1, x2, y2].
        padding: Fractional padding around the box (default 15%).

    Returns:
        Cropped BGR sub-image. May be empty if bbox is invalid.
    """
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


def eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio (EAR) for blink detection.

    Args:
        eye_landmarks: Array of shape (6, 2) with the 6 eye keypoints:
            [P1, P2, P3, P4, P5, P6] where P1-P4 are horizontal,
            P2-P6 and P3-P5 are vertical pairs.

    Returns:
        EAR value (lower = eye more closed).
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return float((A + B) / (2.0 * C + 1e-6))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Both vectors are L2-normalized before dot product.

    Returns:
        Similarity score in [-1, 1]. Higher = more similar.
    """
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


# ─── Response Builders (Flask) ────────────────────────────────────────────────


def success_response(data: dict, status: int = 200):
    """
    Build a standardised JSON success response.

    Schema: {"status": "success", "data": {...}}
    """
    from flask import jsonify

    return jsonify({"status": "success", "data": data}), status


def error_response(message: str, status: int = 400):
    """
    Build a standardised JSON error response.

    Schema: {"status": "error", "message": "..."}
    """
    from flask import jsonify

    return jsonify({"status": "error", "message": message}), status


# ─── Timing Decorator ─────────────────────────────────────────────────────────


def timed(fn: Callable) -> Callable:
    """
    Decorator that logs function execution time in milliseconds.

    Uses DEBUG level logging under the calling module's logger.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.debug("%s completed in %.1f ms", fn.__qualname__, elapsed_ms)
        return result

    return wrapper


# ─── Logging Setup ────────────────────────────────────────────────────────────


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a standard format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

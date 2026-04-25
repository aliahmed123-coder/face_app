"""
models/face_quality.py

Face Quality Assessment Model — per-face quality scoring.

Evaluated dimensions per face:
  • Resolution          — is the face large enough to be reliably recognised?
  • Sharpness           — Laplacian variance (blur detection)
  • Brightness          — mean pixel intensity (over- / under-exposure)
  • Contrast            — RMS contrast
  • Occlusion           — proxy via landmark visibility / edge density
  • Pose deviation      — yaw / pitch / roll from frontal
  • Symmetry            — left-right face symmetry score

Each dimension produces a sub-score in [0, 1].
A weighted average gives an overall quality score.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    QUALITY_MAX_BRIGHTNESS,
    QUALITY_MAX_OCCLUSION,
    QUALITY_MIN_BRIGHTNESS,
    QUALITY_MIN_RESOLUTION,
    QUALITY_MIN_SHARPNESS,
    QUALITY_POSE_THRESHOLD,
)
from utils.helpers import crop_face, timed

logger = logging.getLogger(__name__)


# ─── Per-Dimension Scorers ────────────────────────────────────────────────────

def _score_resolution(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    h, w = face_bgr.shape[:2]
    min_h, min_w = QUALITY_MIN_RESOLUTION
    score = min(1.0, (h / min_h) * 0.5 + (w / min_w) * 0.5)
    return round(score, 4), {"width": w, "height": h, "min": QUALITY_MIN_RESOLUTION}


def _score_sharpness(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = min(1.0, lap_var / (QUALITY_MIN_SHARPNESS * 4))
    return round(score, 4), {"laplacian_variance": round(lap_var, 2)}


def _score_brightness(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    lo, hi = QUALITY_MIN_BRIGHTNESS, QUALITY_MAX_BRIGHTNESS
    if mean_brightness < lo:
        score = mean_brightness / lo
    elif mean_brightness > hi:
        score = 1.0 - (mean_brightness - hi) / (255.0 - hi)
    else:
        score = 1.0
    return round(max(0.0, score), 4), {"mean_brightness": round(mean_brightness, 2)}


def _score_contrast(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rms_contrast = float(gray.std())
    score = min(1.0, rms_contrast / 60.0)
    return round(score, 4), {"rms_contrast": round(rms_contrast, 2)}


def _score_occlusion(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """
    Proxy occlusion metric: edge density in the lower-face region.
    A heavily occluded mouth/nose region (e.g. mask) shows fewer edges.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Bottom 40% of face (mouth/nose area)
    roi = gray[int(h * 0.55):h, :]
    edges = cv2.Canny(roi, 50, 150)
    density = float(edges.mean()) / 255.0
    # Higher edge density → less occlusion
    # Typical unoccluded face: density ~ 0.10–0.20
    score = min(1.0, density / 0.12)
    occlusion_estimate = max(0.0, 1.0 - score)
    return round(score, 4), {"occlusion_estimate": round(occlusion_estimate, 4)}


def _score_pose(head_pose: Optional[Dict]) -> Tuple[float, Dict]:
    """Score from head pose dict (pitch, yaw, roll)."""
    if head_pose is None:
        return 0.8, {}  # Unknown pose → mild penalty
    yaw   = abs(head_pose.get("yaw",   0.0))
    pitch = abs(head_pose.get("pitch", 0.0))
    roll  = abs(head_pose.get("roll",  0.0))
    thr = QUALITY_POSE_THRESHOLD
    # Score decreases linearly from 1 (frontal) to 0 (90° off)
    yaw_s   = max(0.0, 1.0 - yaw   / 90.0)
    pitch_s = max(0.0, 1.0 - pitch / 90.0)
    roll_s  = max(0.0, 1.0 - roll  / 90.0)
    score = (yaw_s * 0.5 + pitch_s * 0.35 + roll_s * 0.15)
    return round(score, 4), {"yaw": yaw, "pitch": pitch, "roll": roll}


def _score_symmetry(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """
    Left-right symmetry via normalised cross-correlation of mirrored halves.
    Real frontal faces are highly symmetric.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    half = w // 2
    left  = gray[:, :half]
    right = gray[:, half:half + half]
    right_flip = right[:, ::-1]
    # Resize if slightly different widths
    min_w = min(left.shape[1], right_flip.shape[1])
    left  = left[:, :min_w]
    right_flip = right_flip[:, :min_w]

    norm = np.sqrt((left ** 2).sum() * (right_flip ** 2).sum() + 1e-8)
    corr = float((left * right_flip).sum() / norm)
    score = max(0.0, corr)
    return round(score, 4), {"symmetry_correlation": round(corr, 4)}


# ─── Face Quality Model ───────────────────────────────────────────────────────

class FaceQualityModel:
    """
    Compute a quality score for each individual face independently.
    All faces in an image are scored separately and returned as a list.

    Quality result schema (per face):
        {
            "bbox":         [x1, y1, x2, y2],
            "overall_score": float,           # weighted average [0, 1]
            "grade":        str,              # "excellent" | "good" | "fair" | "poor"
            "accept":       bool,             # whether quality meets threshold
            "dimensions": {
                "resolution":  {"score": float, ...details},
                "sharpness":   {"score": float, ...details},
                "brightness":  {"score": float, ...details},
                "contrast":    {"score": float, ...details},
                "occlusion":   {"score": float, ...details},
                "pose":        {"score": float, ...details},
                "symmetry":    {"score": float, ...details}
            },
            "issues": List[str]               # human-readable quality problems
        }
    """

    _WEIGHTS = {
        "resolution": 0.25,
        "sharpness":  0.20,
        "brightness": 0.15,
        "contrast":   0.10,
        "occlusion":  0.15,
        "pose":       0.10,
        "symmetry":   0.05,
    }

    _GRADE_THRESHOLDS = {
        "excellent": 0.85,
        "good":      0.70,
        "fair":      0.50,
    }

    _ACCEPT_THRESHOLD = 0.55

    def __init__(self):
        logger.info("FaceQuality: model initialised (dimension-weighted scorer)")

    @timed
    def score_face(
        self,
        face_bgr: np.ndarray,
        bbox: Optional[List[int]] = None,
        head_pose: Optional[Dict] = None,
    ) -> Dict:
        """Compute quality score for a single pre-cropped face image."""

        dims = {}
        scorers = {
            "resolution": lambda: _score_resolution(face_bgr),
            "sharpness":  lambda: _score_sharpness(face_bgr),
            "brightness": lambda: _score_brightness(face_bgr),
            "contrast":   lambda: _score_contrast(face_bgr),
            "occlusion":  lambda: _score_occlusion(face_bgr),
            "pose":       lambda: _score_pose(head_pose),
            "symmetry":   lambda: _score_symmetry(face_bgr),
        }

        for dim_name, scorer in scorers.items():
            try:
                score, detail = scorer()
                dims[dim_name] = {"score": score, **detail}
            except Exception as exc:
                logger.warning("Quality dim '%s' failed: %s", dim_name, exc)
                dims[dim_name] = {"score": 0.5}

        # Weighted average
        overall = sum(
            dims[d]["score"] * self._WEIGHTS[d] for d in self._WEIGHTS
        )
        overall = round(overall, 4)

        grade = "poor"
        for g, thr in self._GRADE_THRESHOLDS.items():
            if overall >= thr:
                grade = g
                break

        issues = self._collect_issues(dims)

        return {
            "bbox": bbox or [],
            "overall_score": overall,
            "grade": grade,
            "accept": overall >= self._ACCEPT_THRESHOLD,
            "dimensions": dims,
            "issues": issues,
        }

    @timed
    def run(
        self,
        image: np.ndarray,
        face_detections: List[Dict],
        head_poses: Optional[List[Optional[Dict]]] = None,
    ) -> List[Dict]:
        """
        Score each detected face independently.

        `face_detections`: list of dicts with "bbox" key (from recogniser / detector).
        `head_poses`: optional list of head pose dicts aligned with face_detections.

        Returns a list of quality result dicts, one per face, preserving all
        original detection fields.
        """
        results = []
        for i, det in enumerate(face_detections):
            bbox = det.get("bbox", [])
            face_crop = crop_face(image, bbox) if bbox else image
            pose = None
            if head_poses and i < len(head_poses):
                pose = head_poses[i]

            if face_crop.size == 0:
                quality = {
                    "bbox": bbox,
                    "overall_score": 0.0,
                    "grade": "poor",
                    "accept": False,
                    "dimensions": {},
                    "issues": ["Face crop empty"],
                }
            else:
                quality = self.score_face(face_crop, bbox=bbox, head_pose=pose)

            results.append({**det, "quality": quality})

        return results

    @staticmethod
    def _collect_issues(dims: Dict) -> List[str]:
        issues = []
        checks = [
            ("resolution", 0.5,  "Face too small"),
            ("sharpness",  0.4,  "Image blurry"),
            ("brightness", 0.35, "Poor exposure"),
            ("contrast",   0.3,  "Low contrast"),
            ("occlusion",  0.5,  "Face partially occluded"),
            ("pose",       0.5,  "Head pose too extreme"),
            ("symmetry",   0.35, "Asymmetric face crop"),
        ]
        for dim, thr, msg in checks:
            if dims.get(dim, {}).get("score", 1.0) < thr:
                issues.append(msg)
        return issues
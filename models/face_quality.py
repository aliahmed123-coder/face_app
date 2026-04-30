"""
models/face_quality.py

Face Quality Assessment — per-face multi-dimensional quality scoring.

Evaluated dimensions (each scored [0, 1]):
    1. Resolution   — face crop size vs minimum requirement
    2. Sharpness    — Laplacian variance (blur detection)
    3. Brightness   — mean pixel intensity (over-/under-exposure)
    4. Contrast     — RMS contrast
    5. Occlusion    — edge density proxy in lower face region
    6. Pose         — yaw/pitch/roll deviation from frontal
    7. Symmetry     — left-right cross-correlation

A weighted average produces an overall quality score and grade.

Usage:
    model = FaceQualityModel()
    result = model.score_face(face_crop)
    # result = {"overall_score": 0.82, "grade": "good", "accept": True, "issues": []}
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    QUALITY_ACCEPT_THRESHOLD,
    QUALITY_MAX_BRIGHTNESS,
    QUALITY_MIN_BRIGHTNESS,
    QUALITY_MIN_RESOLUTION,
    QUALITY_MIN_SHARPNESS,
    QUALITY_POSE_THRESHOLD,
    QUALITY_WEIGHTS,
)
from utils.helpers import crop_face, timed

logger = logging.getLogger(__name__)


# ─── Per-Dimension Scoring Functions ──────────────────────────────────────────


def _score_resolution(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """Score based on face crop dimensions vs minimum requirement."""
    h, w = face_bgr.shape[:2]
    min_h, min_w = QUALITY_MIN_RESOLUTION
    score = min(1.0, (h / min_h) * 0.5 + (w / min_w) * 0.5)
    return round(score, 4), {"width": w, "height": h, "min": list(QUALITY_MIN_RESOLUTION)}


def _score_sharpness(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """Score based on Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    score = min(1.0, lap_var / (QUALITY_MIN_SHARPNESS * 4))
    return round(score, 4), {"laplacian_variance": round(lap_var, 2)}


def _score_brightness(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """Score based on mean brightness (penalises too dark or too bright)."""
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
    """Score based on RMS contrast (standard deviation of pixel intensities)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rms_contrast = float(gray.std())
    score = min(1.0, rms_contrast / 60.0)
    return round(score, 4), {"rms_contrast": round(rms_contrast, 2)}


def _score_occlusion(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """
    Proxy occlusion score based on edge density in the lower face.

    Lower face (mouth/nose region) with fewer edges suggests occlusion
    (e.g., face mask, hand covering mouth).
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    # Bottom 40% of face (mouth/nose area)
    roi = gray[int(h * 0.55) : h, :]
    edges = cv2.Canny(roi, 50, 150)
    density = float(edges.mean()) / 255.0
    # Higher edge density = less occlusion
    score = min(1.0, density / 0.12)
    occlusion_estimate = max(0.0, 1.0 - score)
    return round(score, 4), {"occlusion_estimate": round(occlusion_estimate, 4)}


def _score_pose(head_pose: Optional[Dict]) -> Tuple[float, Dict]:
    """
    Score based on head pose deviation from frontal.

    Penalises large yaw/pitch/roll angles linearly.
    """
    if head_pose is None:
        return 0.8, {}  # Unknown pose -> mild penalty

    yaw = abs(head_pose.get("yaw", 0.0))
    pitch = abs(head_pose.get("pitch", 0.0))
    roll = abs(head_pose.get("roll", 0.0))

    yaw_s = max(0.0, 1.0 - yaw / 90.0)
    pitch_s = max(0.0, 1.0 - pitch / 90.0)
    roll_s = max(0.0, 1.0 - roll / 90.0)
    score = yaw_s * 0.5 + pitch_s * 0.35 + roll_s * 0.15

    return round(score, 4), {"yaw": yaw, "pitch": pitch, "roll": roll}


def _score_symmetry(face_bgr: np.ndarray) -> Tuple[float, Dict]:
    """
    Score based on left-right symmetry via normalised cross-correlation.

    Frontal real faces exhibit high bilateral symmetry.
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    half = w // 2

    left = gray[:, :half]
    right = gray[:, half : half + half]
    right_flip = right[:, ::-1]

    # Handle slight width mismatch
    min_w = min(left.shape[1], right_flip.shape[1])
    left = left[:, :min_w]
    right_flip = right_flip[:, :min_w]

    norm = np.sqrt((left ** 2).sum() * (right_flip ** 2).sum() + 1e-8)
    corr = float((left * right_flip).sum() / norm)
    score = max(0.0, corr)
    return round(score, 4), {"symmetry_correlation": round(corr, 4)}


# ─── Face Quality Model ───────────────────────────────────────────────────────


class FaceQualityModel:
    """
    Multi-dimensional face quality scorer.

    Each face is scored independently across 7 dimensions. A weighted average
    yields an overall quality score, grade, and accept/reject decision.

    Quality result schema:
        {
            "bbox":          [x1, y1, x2, y2],
            "overall_score": float,               # weighted average [0, 1]
            "grade":         str,                 # "excellent" | "good" | "fair" | "poor"
            "accept":        bool,                # meets minimum threshold
            "dimensions": {
                "resolution": {"score": float, ...},
                "sharpness":  {"score": float, ...},
                ...
            },
            "issues":        List[str]            # human-readable quality problems
        }
    """

    _GRADE_THRESHOLDS = {
        "excellent": 0.85,
        "good": 0.70,
        "fair": 0.50,
    }

    _ISSUE_CHECKS = [
        ("resolution", 0.5, "Face too small"),
        ("sharpness", 0.4, "Image blurry"),
        ("brightness", 0.35, "Poor exposure"),
        ("contrast", 0.3, "Low contrast"),
        ("occlusion", 0.5, "Face partially occluded"),
        ("pose", 0.5, "Head pose too extreme"),
        ("symmetry", 0.35, "Asymmetric face crop"),
    ]

    def __init__(self):
        logger.info("FaceQuality: model initialised (7-dimension weighted scorer)")

    @timed
    def score_face(
        self,
        face_bgr: np.ndarray,
        bbox: Optional[List[int]] = None,
        head_pose: Optional[Dict] = None,
    ) -> Dict:
        """
        Compute quality score for a single pre-cropped face image.

        Args:
            face_bgr: BGR face crop.
            bbox: Original bounding box (for reference in output).
            head_pose: Dict with pitch/yaw/roll (from liveness module).

        Returns:
            Quality assessment dict with overall_score, grade, accept, dimensions, issues.
        """
        dims = {}
        scorers = {
            "resolution": lambda: _score_resolution(face_bgr),
            "sharpness": lambda: _score_sharpness(face_bgr),
            "brightness": lambda: _score_brightness(face_bgr),
            "contrast": lambda: _score_contrast(face_bgr),
            "occlusion": lambda: _score_occlusion(face_bgr),
            "pose": lambda: _score_pose(head_pose),
            "symmetry": lambda: _score_symmetry(face_bgr),
        }

        for dim_name, scorer in scorers.items():
            try:
                score, detail = scorer()
                dims[dim_name] = {"score": score, **detail}
            except Exception as exc:
                logger.warning("Quality dimension '%s' failed: %s", dim_name, exc)
                dims[dim_name] = {"score": 0.5}

        # Weighted average
        overall = sum(
            dims[d]["score"] * QUALITY_WEIGHTS[d] for d in QUALITY_WEIGHTS
        )
        overall = round(overall, 4)

        # Determine grade
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
            "accept": overall >= QUALITY_ACCEPT_THRESHOLD,
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
        Score quality for each detected face in an image.

        Args:
            image: Full BGR image.
            face_detections: List of dicts with "bbox" key.
            head_poses: Optional list of head pose dicts aligned with detections.

        Returns:
            List of detection dicts enriched with a "quality" field.
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

    @classmethod
    def _collect_issues(cls, dims: Dict) -> List[str]:
        """Generate human-readable issue descriptions from dimension scores."""
        issues = []
        for dim, threshold, message in cls._ISSUE_CHECKS:
            if dims.get(dim, {}).get("score", 1.0) < threshold:
                issues.append(message)
        return issues

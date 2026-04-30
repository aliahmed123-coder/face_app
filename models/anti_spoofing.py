"""
models/anti_spoofing.py

Anti-Spoofing Model — Texture (LBP) + Frequency (FFT) + optional CNN (MiniFASNet ONNX).

Detects presentation attacks including:
    - Print attacks (paper photos)
    - Replay attacks (screen display)
    - 3D mask attacks

Responsibilities:
    - Classify each face crop as live or spoof
    - Return probability and attack type classification
    - Support both CNN-based (when weights available) and heuristic-based analysis

Usage:
    model = AntiSpoofingModel()
    result = model.predict_face(face_crop)
    # result = {"is_live": True, "live_prob": 0.92, "spoof_type": "live", "method": "heuristic"}
"""

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    SPOOF_FFT_WEIGHT,
    SPOOF_INPUT_SIZE,
    SPOOF_LBP_WEIGHT,
    SPOOF_MODEL_PATH,
    SPOOF_THRESHOLD,
)
from utils.helpers import crop_face, timed

logger = logging.getLogger(__name__)


# ─── LBP Texture Analysis ────────────────────────────────────────────────────


def _compute_lbp_histogram(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Compute a Uniform Local Binary Pattern histogram.

    LBP encodes local texture patterns. Real faces have richer texture
    variability than printed or screen-displayed images.

    Args:
        gray: Grayscale image.
        radius: Radius for sampling neighbors.
        n_points: Number of circular sample points.

    Returns:
        Normalised 256-bin histogram.
    """
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.float32)

    for p in range(n_points):
        angle = 2 * np.pi * p / n_points
        dx = int(round(radius * np.cos(angle)))
        dy = int(round(-radius * np.sin(angle)))
        shifted = np.roll(np.roll(gray.astype(np.float32), dy, axis=0), dx, axis=1)
        lbp += (shifted >= gray.astype(np.float32)).astype(np.float32) * (2 ** p)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return hist


def _texture_liveness_score(face_bgr: np.ndarray) -> float:
    """
    Compute a liveness score from LBP texture entropy.

    Real faces exhibit higher LBP entropy (5-7 bits) compared to
    print attacks (3-4 bits) due to richer micro-texture.

    Args:
        face_bgr: BGR face crop.

    Returns:
        Score in [0, 1] where 1 = likely live.
    """
    face_resized = cv2.resize(face_bgr, (64, 64))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    hist = _compute_lbp_histogram(gray)
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    # Normalise: real face entropy ~ 5-7 bits; paper ~ 3-4 bits
    score = float(np.clip((entropy - 3.0) / 4.0, 0.0, 1.0))
    return score


# ─── Frequency Analysis (FFT) ────────────────────────────────────────────────


def _frequency_liveness_score(face_bgr: np.ndarray) -> float:
    """
    Compute a liveness score from frequency domain analysis.

    Real faces contain richer high-frequency content than reproductions
    which lose detail through printing or screen display.

    Args:
        face_bgr: BGR face crop.

    Returns:
        Score in [0, 1] where 1 = likely live.
    """
    gray = cv2.cvtColor(
        cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    centre = magnitude[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    hf_ratio = 1.0 - (centre.sum() / (magnitude.sum() + 1e-8))
    return float(np.clip(hf_ratio * 3.0, 0.0, 1.0))


# ─── Anti-Spoofing Model ─────────────────────────────────────────────────────


class AntiSpoofingModel:
    """
    Two-stage anti-spoofing classifier.

    Stage 1 (primary): CNN — MiniFASNet ONNX model (if weights are available).
    Stage 2 (fallback): Ensemble of LBP texture + FFT frequency heuristics.

    Output schema per face:
        {
            "is_live":    bool,       # Whether the face passes the liveness threshold
            "live_prob":  float,      # Probability of being a real face [0, 1]
            "spoof_type": str,        # "live" | "print" | "replay" | "mask"
            "method":     str         # "cnn" | "heuristic"
        }

    Args:
        model_path: Path to MiniFASNet ONNX weights. None = heuristic only.
    """

    _SPOOF_LABELS = {0: "live", 1: "print", 2: "replay", 3: "mask"}

    def __init__(self, model_path: Optional[str] = None):
        self._ort_session = None
        self._input_name: Optional[str] = None
        self._model_path = model_path or SPOOF_MODEL_PATH
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load the ONNX CNN model."""
        if self._model_path and os.path.exists(self._model_path):
            try:
                import onnxruntime as ort

                self._ort_session = ort.InferenceSession(
                    self._model_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self._input_name = self._ort_session.get_inputs()[0].name
                logger.info(
                    "AntiSpoofing: MiniFASNet ONNX loaded from %s", self._model_path
                )
            except Exception as exc:
                logger.warning(
                    "AntiSpoofing: ONNX load failed (%s), using heuristics", exc
                )
        else:
            logger.info(
                "AntiSpoofing: no ONNX weights at %s - using texture/frequency heuristics",
                self._model_path,
            )

    @timed
    def predict_face(self, face_bgr: np.ndarray) -> Dict:
        """
        Classify a single pre-cropped face as live or spoof.

        Args:
            face_bgr: BGR face crop (any size, will be resized internally).

        Returns:
            Dict with is_live, live_prob, spoof_type, method.
        """
        if self._ort_session is not None:
            return self._cnn_predict(face_bgr)
        return self._heuristic_predict(face_bgr)

    def _cnn_predict(self, face_bgr: np.ndarray) -> Dict:
        """Run MiniFASNet ONNX inference."""
        inp = cv2.resize(face_bgr, SPOOF_INPUT_SIZE).astype(np.float32)
        inp = inp[..., ::-1]  # BGR -> RGB
        inp = (inp / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis].astype(np.float32)  # NCHW

        logits = self._ort_session.run(None, {self._input_name: inp})[0][0]
        probs = self._softmax(logits)
        live_prob = float(probs[0])
        class_id = int(np.argmax(probs))

        return {
            "is_live": live_prob >= SPOOF_THRESHOLD,
            "live_prob": round(live_prob, 4),
            "spoof_type": self._SPOOF_LABELS.get(class_id, "unknown"),
            "method": "cnn",
        }

    def _heuristic_predict(self, face_bgr: np.ndarray) -> Dict:
        """Ensemble of LBP texture + FFT frequency analysis."""
        tex_score = _texture_liveness_score(face_bgr)
        freq_score = _frequency_liveness_score(face_bgr)
        live_prob = SPOOF_LBP_WEIGHT * tex_score + SPOOF_FFT_WEIGHT * freq_score
        is_live = live_prob >= SPOOF_THRESHOLD

        spoof_type = "live"
        if not is_live:
            if tex_score < 0.4:
                spoof_type = "print"
            else:
                spoof_type = "replay"

        return {
            "is_live": is_live,
            "live_prob": round(live_prob, 4),
            "spoof_type": spoof_type,
            "method": "heuristic",
        }

    @timed
    def run(self, image: np.ndarray, face_detections: List[Dict]) -> List[Dict]:
        """
        Run anti-spoofing on all detected faces in an image.

        Args:
            image: Full BGR image.
            face_detections: List of dicts, each with at least a "bbox" key.

        Returns:
            Same list with a "spoof" field merged into each dict.
        """
        results = []
        for det in face_detections:
            face_crop = crop_face(image, det["bbox"])
            if face_crop.size == 0:
                spoof_result = {
                    "is_live": False,
                    "live_prob": 0.0,
                    "spoof_type": "unknown",
                    "method": "none",
                }
            else:
                spoof_result = self.predict_face(face_crop)

            results.append({**det, "spoof": spoof_result})

        return results

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()

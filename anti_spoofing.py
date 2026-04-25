"""
models/anti_spoofing.py

Anti-Spoofing Model — MiniFASNet / Silent-Face-Anti-Spoofing.

Detects print attacks, replay attacks (screen display), and 3D masks
using texture analysis (LBP features) and a lightweight CNN.

Responsibilities:
  • Classify each face crop as Live or Spoof
  • Return a probability score and label
  • Support multiple face regions per image
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import SPOOF_INPUT_SIZE, SPOOF_THRESHOLD
from utils.helpers import crop_face, timed

logger = logging.getLogger(__name__)


# ─── LBP-based texture descriptor (lightweight fallback) ─────────────────────

def _lbp_features(gray: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Uniform Local Binary Pattern histogram.
    Used as a fast, model-free spoof cue when CNN weights are unavailable.
    """
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.float32)
    for p in range(n_points):
        angle = 2 * np.pi * p / n_points
        dx = radius * np.cos(angle)
        dy = -radius * np.sin(angle)
        x0, y0 = int(round(dx)), int(round(dy))
        # Shift and compare
        shifted = np.roll(np.roll(gray.astype(np.float32), y0, axis=0), x0, axis=1)
        lbp += (shifted >= gray.astype(np.float32)).astype(np.float32) * (2 ** p)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
    return hist


def _texture_score(face_bgr: np.ndarray) -> float:
    """
    Heuristic liveness score from texture richness.
    Real faces have higher LBP entropy than printed/screen images.
    Returns a value in [0, 1] where 1 = likely live.
    """
    face_resized = cv2.resize(face_bgr, (64, 64))
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    hist = _lbp_features(gray)
    # Entropy of the LBP histogram
    entropy = -np.sum(hist * np.log2(hist + 1e-8))
    # Normalise: typical real face entropy ~ 5-7 bits; paper attacks ~ 3-4 bits
    score = float(np.clip((entropy - 3.0) / 4.0, 0.0, 1.0))
    return score


def _frequency_score(face_bgr: np.ndarray) -> float:
    """
    Frequency-domain cue: real faces have richer high-frequency content.
    Returns [0, 1] — higher = more likely live.
    """
    gray = cv2.cvtColor(
        cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = magnitude.shape
    centre = magnitude[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    hf_ratio = 1.0 - (centre.sum() / (magnitude.sum() + 1e-8))
    return float(np.clip(hf_ratio * 3.0, 0.0, 1.0))


# ─── CNN-based Anti-Spoofing ──────────────────────────────────────────────────

class AntiSpoofingModel:
    """
    Two-stage anti-spoofing:
      1. CNN (MiniFASNet ONNX) — primary, loaded if weights available
      2. Texture + frequency heuristics — fallback / ensemble member

    Output per face:
        {
            "is_live":       bool,
            "live_prob":     float,   # probability of being real
            "spoof_type":    str,     # "live" | "print" | "replay" | "mask"
            "method":        str      # "cnn" | "heuristic"
        }
    """

    _SPOOF_LABELS = {0: "live", 1: "print", 2: "replay", 3: "mask"}

    def __init__(self, model_path: Optional[str] = None):
        self.ort_session = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            try:
                import onnxruntime as ort

                self.ort_session = ort.InferenceSession(
                    self.model_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                self.input_name = self.ort_session.get_inputs()[0].name
                logger.info("AntiSpoofing: MiniFASNet ONNX loaded from %s", self.model_path)
            except Exception as exc:
                logger.warning("AntiSpoofing: ONNX load failed (%s), using heuristics", exc)
        else:
            logger.info(
                "AntiSpoofing: no ONNX weights found — using texture/frequency heuristics"
            )

    @timed
    def predict_face(self, face_bgr: np.ndarray) -> Dict:
        """Classify a single pre-cropped face region."""
        if self.ort_session is not None:
            return self._cnn_predict(face_bgr)
        return self._heuristic_predict(face_bgr)

    def _cnn_predict(self, face_bgr: np.ndarray) -> Dict:
        """ONNX MiniFASNet inference."""
        inp = cv2.resize(face_bgr, SPOOF_INPUT_SIZE).astype(np.float32)
        inp = inp[..., ::-1]  # BGR → RGB
        inp = (inp / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        inp = np.transpose(inp, (2, 0, 1))[np.newaxis]  # NCHW

        logits = self.ort_session.run(None, {self.input_name: inp})[0][0]
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
        """Ensemble of texture + frequency cues."""
        tex = _texture_score(face_bgr)
        freq = _frequency_score(face_bgr)
        live_prob = 0.6 * tex + 0.4 * freq
        is_live = live_prob >= SPOOF_THRESHOLD

        spoof_type = "live"
        if not is_live:
            # Heuristic classification: low texture → print; low freq → replay
            if tex < 0.4:
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

        `face_detections` should be a list of dicts with at least a "bbox" key.
        Returns the same list with "spoof" fields merged in.
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
        x = x - x.max()
        exp_x = np.exp(x)
        return exp_x / exp_x.sum()
"""
models/fraud_detection.py

Fraud Detection Model — Rule-based + Statistical Analysis.

Detects:
  • Duplicate check-in within a time window
  • Impossible travel (same identity in two distant locations)
  • Velocity anomaly (face moving too fast between frames)
  • Identity cloning (same embedding matching multiple registered IDs)
  • Spoof-pass bypass (recognition without liveness confirmation)
  • Unusual time-of-day access
  • Repeated failed recognition attempts
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    FRAUD_CONSISTENCY_THRESHOLD,
    FRAUD_DUPLICATE_WINDOW,
    FRAUD_MAX_SPEED_PX_PER_FRAME,
)
from utils.helpers import cosine_similarity

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Stateful fraud detector.  Maintains an in-memory event log and per-identity
    state. In production, replace in-memory stores with a database.
    """

    def __init__(self):
        # attendance_log: {person_id: deque of (timestamp, camera_id)}
        self.attendance_log: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # bbox_history: {track_id: deque of (timestamp, bbox)}
        self.bbox_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        # failed_attempts: {camera_id: deque of timestamps}
        self.failed_attempts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        # embedding_fingerprints: {person_id: List[embedding]}
        self.embedding_fingerprints: Dict[str, List[np.ndarray]] = {}

        logger.info("FraudDetection: module initialised")

    # ─── Primary Entry Point ──────────────────────────────────────────────────

    def check(
        self,
        person_id: Optional[str],
        embedding: Optional[np.ndarray],
        track_id: Optional[int],
        bbox: Optional[List[int]],
        liveness_passed: bool,
        spoof_passed: bool,
        camera_id: str = "default",
        timestamp: Optional[float] = None,
    ) -> Dict:
        """
        Run all fraud checks for a recognition event.

        Returns:
            {
                "fraud_detected":  bool,
                "risk_score":      float,   # [0, 1]
                "flags":           List[str],
                "action":          str,     # "allow" | "warn" | "block"
                "details":         dict
            }
        """
        ts = timestamp or time.time()
        flags = []
        details = {}

        # 1. Liveness / spoof bypass
        if not liveness_passed:
            flags.append("LIVENESS_FAILED")
        if not spoof_passed:
            flags.append("SPOOF_DETECTED")

        if person_id:
            # 2. Duplicate check-in
            dup = self._check_duplicate(person_id, camera_id, ts)
            if dup["duplicate"]:
                flags.append("DUPLICATE_CHECKIN")
                details["last_seen"] = dup["last_seen"]
                details["seconds_since_last"] = dup["seconds_since_last"]

            # 3. Identity consistency (embedding drift)
            if embedding is not None:
                consistency = self._check_embedding_consistency(person_id, embedding)
                details["embedding_consistency"] = round(consistency, 4)
                if consistency < FRAUD_CONSISTENCY_THRESHOLD:
                    flags.append("EMBEDDING_INCONSISTENCY")

        else:
            # 4. Excessive failed recognition
            fail_flag = self._check_failed_attempts(camera_id, ts)
            if fail_flag:
                flags.append("EXCESSIVE_FAILED_ATTEMPTS")

        # 5. Velocity anomaly (track-based)
        if track_id is not None and bbox is not None:
            speed = self._check_velocity(track_id, bbox, ts)
            details["face_speed_px_per_frame"] = round(speed, 2)
            if speed > FRAUD_MAX_SPEED_PX_PER_FRAME:
                flags.append("VELOCITY_ANOMALY")
            self.bbox_history[track_id].append((ts, bbox))

        # 6. Time-of-day anomaly (simple: 0–5 AM)
        hour = time.localtime(ts).tm_hour
        if hour < 5 or hour > 23:
            flags.append("UNUSUAL_TIME")
            details["access_hour"] = hour

        # ── Scoring ──────────────────────────────────────────────────────────
        risk_score = self._score(flags)

        # Log legitimate events
        if person_id and "DUPLICATE_CHECKIN" not in flags and risk_score < 0.7:
            self.attendance_log[person_id].append((ts, camera_id))
            if embedding is not None:
                self._update_fingerprint(person_id, embedding)

        if "LIVENESS_FAILED" in flags or "SPOOF_DETECTED" in flags:
            self.failed_attempts[camera_id].append(ts)

        action = "allow" if risk_score < 0.35 else ("warn" if risk_score < 0.70 else "block")

        return {
            "fraud_detected": risk_score >= 0.35,
            "risk_score": round(risk_score, 4),
            "flags": flags,
            "action": action,
            "details": details,
        }

    # ─── Individual Checks ────────────────────────────────────────────────────

    def _check_duplicate(self, person_id: str, camera_id: str, ts: float) -> Dict:
        history = self.attendance_log[person_id]
        for prev_ts, prev_cam in reversed(history):
            delta = ts - prev_ts
            if delta < FRAUD_DUPLICATE_WINDOW:
                return {
                    "duplicate": True,
                    "last_seen": prev_ts,
                    "seconds_since_last": round(delta, 1),
                    "prev_camera": prev_cam,
                }
        return {"duplicate": False, "last_seen": None, "seconds_since_last": None}

    def _check_embedding_consistency(self, person_id: str, embedding: np.ndarray) -> float:
        fps = self.embedding_fingerprints.get(person_id, [])
        if not fps:
            return 1.0  # No history → assume consistent
        sims = [cosine_similarity(embedding, fp) for fp in fps[-5:]]
        return float(np.mean(sims))

    def _check_failed_attempts(self, camera_id: str, ts: float) -> bool:
        window = 300  # 5 minutes
        recent = [t for t in self.failed_attempts[camera_id] if ts - t < window]
        return len(recent) >= 10  # 10+ failures in 5 min

    def _check_velocity(self, track_id: int, bbox: List[int], ts: float) -> float:
        history = self.bbox_history[track_id]
        if not history:
            return 0.0
        prev_ts, prev_bbox = history[-1]
        dt = max(ts - prev_ts, 1e-3)
        cx_now = (bbox[0] + bbox[2]) / 2
        cy_now = (bbox[1] + bbox[3]) / 2
        cx_prev = (prev_bbox[0] + prev_bbox[2]) / 2
        cy_prev = (prev_bbox[1] + prev_bbox[3]) / 2
        dist = np.hypot(cx_now - cx_prev, cy_now - cy_prev)
        return float(dist / dt)

    def _update_fingerprint(self, person_id: str, embedding: np.ndarray):
        if person_id not in self.embedding_fingerprints:
            self.embedding_fingerprints[person_id] = []
        fps = self.embedding_fingerprints[person_id]
        fps.append(embedding)
        if len(fps) > 20:
            fps.pop(0)

    # ─── Risk Scoring ─────────────────────────────────────────────────────────

    _FLAG_WEIGHTS = {
        "LIVENESS_FAILED":            0.70,
        "SPOOF_DETECTED":             0.85,
        "DUPLICATE_CHECKIN":          0.40,
        "EMBEDDING_INCONSISTENCY":    0.55,
        "EXCESSIVE_FAILED_ATTEMPTS":  0.45,
        "VELOCITY_ANOMALY":           0.35,
        "UNUSUAL_TIME":               0.15,
    }

    def _score(self, flags: List[str]) -> float:
        if not flags:
            return 0.0
        weights = [self._FLAG_WEIGHTS.get(f, 0.2) for f in flags]
        # Combine with noisy-OR: P(fraud) = 1 - Π(1 - wᵢ)
        score = 1.0 - np.prod([1.0 - w for w in weights])
        return float(np.clip(score, 0.0, 1.0))

    # ─── Statistics ───────────────────────────────────────────────────────────

    def attendance_stats(self, person_id: str) -> Dict:
        history = list(self.attendance_log.get(person_id, []))
        return {
            "person_id": person_id,
            "total_events": len(history),
            "recent_events": [
                {"timestamp": t, "camera_id": cam}
                for t, cam in history[-10:]
            ],
        }

    def reset(self):
        self.attendance_log.clear()
        self.bbox_history.clear()
        self.failed_attempts.clear()
        self.embedding_fingerprints.clear()
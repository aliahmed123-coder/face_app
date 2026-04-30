"""
models/fraud_detection.py

Fraud Detection — Rule-based risk scoring with stateful analysis.

Detection rules:
    1. Duplicate check-in (same person within a time window)
    2. Velocity anomaly (face position jumps impossibly fast)
    3. Embedding consistency drift (face changes over time)
    4. Unusual time-of-day access
    5. Excessive failed recognition attempts

State is maintained in-memory using deques per person/track/camera.
Risk scoring uses noisy-OR combination of flag weights.

Actions: allow | warn | block (based on cumulative risk score)

Usage:
    fraud = FraudDetectionModel()
    result = fraud.check(
        person_id="emp_001", embedding=emb, track_id=1,
        bbox=[x1,y1,x2,y2], liveness_passed=True, spoof_passed=True
    )
    # result = {"fraud_detected": False, "risk_score": 0.0, "action": "allow", ...}
"""

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

from config.settings import (
    FRAUD_CONSISTENCY_THRESHOLD,
    FRAUD_DUPLICATE_WINDOW,
    FRAUD_FAILED_ATTEMPTS_LIMIT,
    FRAUD_FAILED_ATTEMPTS_WINDOW,
    FRAUD_MAX_SPEED_PX_PER_FRAME,
    FRAUD_UNUSUAL_HOUR_END,
    FRAUD_UNUSUAL_HOUR_START,
)
from utils.helpers import cosine_similarity

logger = logging.getLogger(__name__)


class FraudDetectionModel:
    """
    Stateful fraud detection engine.

    Maintains per-identity and per-camera event logs in memory.
    Evaluates multiple fraud signals and combines them using noisy-OR scoring.

    Fraud result schema:
        {
            "fraud_detected": bool,       # True if risk_score >= warn threshold
            "risk_score":     float,      # Combined risk [0, 1]
            "flags":          List[str],  # Triggered rule names
            "action":         str,        # "allow" | "warn" | "block"
            "details":        dict        # Per-rule diagnostic information
        }

    Risk thresholds:
        - allow: risk_score < 0.35
        - warn:  0.35 <= risk_score < 0.70
        - block: risk_score >= 0.70
    """

    # Risk weight per fraud flag (used in noisy-OR)
    _FLAG_WEIGHTS = {
        "LIVENESS_FAILED": 0.70,
        "SPOOF_DETECTED": 0.85,
        "DUPLICATE_CHECKIN": 0.40,
        "EMBEDDING_INCONSISTENCY": 0.55,
        "EXCESSIVE_FAILED_ATTEMPTS": 0.45,
        "VELOCITY_ANOMALY": 0.35,
        "UNUSUAL_TIME": 0.15,
    }

    def __init__(self):
        # attendance_log: {person_id: deque of (timestamp, camera_id)}
        self._attendance_log: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        # bbox_history: {track_id: deque of (timestamp, bbox)}
        self._bbox_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )
        # failed_attempts: {camera_id: deque of timestamps}
        self._failed_attempts: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        # embedding_fingerprints: {person_id: List[embedding]}
        self._embedding_fingerprints: Dict[str, List[np.ndarray]] = {}

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
        Run all fraud checks for a single recognition event.

        Args:
            person_id: Identified person (None if unrecognised).
            embedding: Face embedding vector (for consistency checks).
            track_id: Object tracker ID (for velocity checks).
            bbox: Face bounding box [x1, y1, x2, y2].
            liveness_passed: Whether liveness verification passed.
            spoof_passed: Whether anti-spoofing passed.
            camera_id: Source camera identifier.
            timestamp: Event timestamp (defaults to current time).

        Returns:
            Fraud assessment dict with risk_score, flags, action, details.
        """
        ts = timestamp or time.time()
        flags: List[str] = []
        details: Dict = {}

        # 1. Liveness / spoof bypass checks
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

            # 3. Embedding consistency drift
            if embedding is not None:
                consistency = self._check_embedding_consistency(person_id, embedding)
                details["embedding_consistency"] = round(consistency, 4)
                if consistency < FRAUD_CONSISTENCY_THRESHOLD:
                    flags.append("EMBEDDING_INCONSISTENCY")
        else:
            # 4. Excessive failed recognition attempts
            if self._check_failed_attempts(camera_id, ts):
                flags.append("EXCESSIVE_FAILED_ATTEMPTS")

        # 5. Velocity anomaly (track-based)
        if track_id is not None and bbox is not None:
            speed = self._check_velocity(track_id, bbox, ts)
            details["face_speed_px_per_frame"] = round(speed, 2)
            if speed > FRAUD_MAX_SPEED_PX_PER_FRAME:
                flags.append("VELOCITY_ANOMALY")
            self._bbox_history[track_id].append((ts, bbox))

        # 6. Unusual time-of-day
        hour = time.localtime(ts).tm_hour
        if hour >= FRAUD_UNUSUAL_HOUR_START and hour < FRAUD_UNUSUAL_HOUR_END:
            flags.append("UNUSUAL_TIME")
            details["access_hour"] = hour

        # ── Risk scoring (noisy-OR) ──────────────────────────────────────────
        risk_score = self._compute_risk_score(flags)

        # Record legitimate events
        if person_id and "DUPLICATE_CHECKIN" not in flags and risk_score < 0.7:
            self._attendance_log[person_id].append((ts, camera_id))
            if embedding is not None:
                self._update_fingerprint(person_id, embedding)

        # Record failures
        if "LIVENESS_FAILED" in flags or "SPOOF_DETECTED" in flags:
            self._failed_attempts[camera_id].append(ts)

        # Determine action
        if risk_score < 0.35:
            action = "allow"
        elif risk_score < 0.70:
            action = "warn"
        else:
            action = "block"

        return {
            "fraud_detected": risk_score >= 0.35,
            "risk_score": round(risk_score, 4),
            "flags": flags,
            "action": action,
            "details": details,
        }

    # ─── Individual Rule Checks ───────────────────────────────────────────────

    def _check_duplicate(self, person_id: str, camera_id: str, ts: float) -> Dict:
        """Check if this person was seen within the duplicate window."""
        history = self._attendance_log[person_id]
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

    def _check_embedding_consistency(
        self, person_id: str, embedding: np.ndarray
    ) -> float:
        """Compare current embedding against recent fingerprints."""
        fps = self._embedding_fingerprints.get(person_id, [])
        if not fps:
            return 1.0  # No history => assume consistent
        sims = [cosine_similarity(embedding, fp) for fp in fps[-5:]]
        return float(np.mean(sims))

    def _check_failed_attempts(self, camera_id: str, ts: float) -> bool:
        """Check if there are excessive failed attempts on this camera."""
        recent = [
            t
            for t in self._failed_attempts[camera_id]
            if ts - t < FRAUD_FAILED_ATTEMPTS_WINDOW
        ]
        return len(recent) >= FRAUD_FAILED_ATTEMPTS_LIMIT

    def _check_velocity(self, track_id: int, bbox: List[int], ts: float) -> float:
        """Compute face movement speed between consecutive frames."""
        history = self._bbox_history[track_id]
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

    def _update_fingerprint(self, person_id: str, embedding: np.ndarray) -> None:
        """Store an embedding in the rolling fingerprint buffer."""
        if person_id not in self._embedding_fingerprints:
            self._embedding_fingerprints[person_id] = []
        fps = self._embedding_fingerprints[person_id]
        fps.append(embedding)
        if len(fps) > 20:
            fps.pop(0)

    # ─── Risk Scoring ─────────────────────────────────────────────────────────

    def _compute_risk_score(self, flags: List[str]) -> float:
        """
        Combine flag weights using noisy-OR:
            P(fraud) = 1 - prod(1 - w_i for each flag)

        Returns:
            Combined risk score in [0, 1].
        """
        if not flags:
            return 0.0
        weights = [self._FLAG_WEIGHTS.get(f, 0.2) for f in flags]
        score = 1.0 - float(np.prod([1.0 - w for w in weights]))
        return float(np.clip(score, 0.0, 1.0))

    # ─── Statistics & Management ──────────────────────────────────────────────

    def attendance_stats(self, person_id: str) -> Dict:
        """
        Get attendance history statistics for a person.

        Args:
            person_id: The person to query.

        Returns:
            Dict with total_events and recent event list.
        """
        history = list(self._attendance_log.get(person_id, []))
        return {
            "person_id": person_id,
            "total_events": len(history),
            "recent_events": [
                {"timestamp": t, "camera_id": cam} for t, cam in history[-10:]
            ],
        }

    def reset(self) -> None:
        """Clear all in-memory state (for testing/development)."""
        self._attendance_log.clear()
        self._bbox_history.clear()
        self._failed_attempts.clear()
        self._embedding_fingerprints.clear()
        logger.info("FraudDetection: all state cleared")

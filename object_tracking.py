"""
models/object_tracking.py

Object Tracking Model — SORT (Simple Online and Realtime Tracking)
with optional ByteTrack upgrade when ultralytics >= 8.1 is available.

Responsibilities:
  • Assign persistent track IDs to detections across frames
  • Maintain per-track state (bbox history, age, face_id association)
  • Support track lifecycle: new → confirmed → lost → deleted
  • Associate face recognition results with tracks
"""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    TRACKER_IOU_THRESHOLD,
    TRACKER_MAX_AGE,
    TRACKER_MIN_HITS,
)
from utils.helpers import timed

logger = logging.getLogger(__name__)


# ─── Kalman-filter-based SORT implementation ──────────────────────────────────

class KalmanTrack:
    """Single object track with a simple constant-velocity Kalman filter."""

    _id_counter = 0

    def __init__(self, bbox: List[int]):
        KalmanTrack._id_counter += 1
        self.track_id = KalmanTrack._id_counter
        self.hits = 1
        self.no_match_count = 0
        self.time_since_update = 0
        self.history: List[List[int]] = [bbox]
        self.face_id: Optional[str] = None
        self.face_name: Optional[str] = None
        self.face_confidence: float = 0.0
        self.created_at = time.time()
        self.last_seen = time.time()

        # State: [cx, cy, s, r, vx, vy, vs]
        # where s = scale (area), r = aspect ratio (w/h)
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / float(y2 - y1 + 1e-6)

        try:
            from filterpy.kalman import KalmanFilter

            kf = KalmanFilter(dim_x=7, dim_z=4)
            kf.F = np.array([
                [1,0,0,0,1,0,0],
                [0,1,0,0,0,1,0],
                [0,0,1,0,0,0,1],
                [0,0,0,1,0,0,0],
                [0,0,0,0,1,0,0],
                [0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1],
            ], dtype=float)
            kf.H = np.array([
                [1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [0,0,1,0,0,0,0],
                [0,0,0,1,0,0,0],
            ], dtype=float)
            kf.R[2:, 2:] *= 10.
            kf.P[4:, 4:] *= 1000.
            kf.P *= 10.
            kf.Q[-1, -1] *= 0.01
            kf.Q[4:, 4:] *= 0.01
            kf.x[:4] = np.array([[cx], [cy], [s], [r]])
            self._kf = kf
            self._use_kf = True
        except ImportError:
            self._kf = None
            self._use_kf = False
            self._state = np.array([cx, cy, s, r, 0., 0., 0.])

    @property
    def bbox(self) -> List[int]:
        return self.history[-1]

    def predict(self):
        if self._use_kf:
            if self._kf.x[6] + self._kf.x[2] <= 0:
                self._kf.x[6] *= 0.0
            self._kf.predict()
            cx, cy, s, r = self._kf.x[:4, 0]
        else:
            self._state[:2] += self._state[4:6]
            cx, cy, s, r = self._state[:4]

        w = np.sqrt(abs(s) * abs(r))
        h = abs(s) / (w + 1e-6)
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        return [x1, y1, x2, y2]

    def update(self, bbox: List[int]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = (x2 - x1) * (y2 - y1)
        r = (x2 - x1) / float(y2 - y1 + 1e-6)

        if self._use_kf:
            self._kf.update(np.array([[cx], [cy], [s], [r]]))
        else:
            self._state[:4] = [cx, cy, s, r]

        self.history.append(bbox)
        if len(self.history) > 50:
            self.history.pop(0)
        self.hits += 1
        self.time_since_update = 0
        self.last_seen = time.time()


def _iou(b1, b2) -> float:
    xa1, ya1, xa2, ya2 = b1
    xb1, yb1, xb2, yb2 = b2
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / (union + 1e-6)


def _greedy_match(tracks, detections, iou_thresh):
    """Greedy IoU matching (no scipy dependency required)."""
    matched = {}
    used_det = set()
    for i, track in enumerate(tracks):
        best_iou = iou_thresh
        best_j = -1
        pred_bbox = track.predict()
        for j, det in enumerate(detections):
            if j in used_det:
                continue
            score = _iou(pred_bbox, det["bbox"])
            if score > best_iou:
                best_iou = score
                best_j = j
        if best_j >= 0:
            matched[i] = best_j
            used_det.add(best_j)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched]
    unmatched_dets = [j for j in range(len(detections)) if j not in used_det]
    return matched, unmatched_tracks, unmatched_dets


# ─── Main Tracker ─────────────────────────────────────────────────────────────

class ObjectTrackingModel:
    """
    Frame-by-frame SORT tracker.

    Usage:
        tracker = ObjectTrackingModel()
        for frame in video_frames:
            detections = detector.detect_persons(frame)
            tracks = tracker.update(detections)
    """

    def __init__(
        self,
        max_age: int = TRACKER_MAX_AGE,
        min_hits: int = TRACKER_MIN_HITS,
        iou_threshold: float = TRACKER_IOU_THRESHOLD,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[KalmanTrack] = []
        self.frame_count = 0
        KalmanTrack._id_counter = 0
        logger.info("ObjectTracking: SORT tracker initialised")

    @timed
    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Feed detections for the current frame, return active tracks.

        Returns:
            List of track dicts:
                {
                    "track_id": int,
                    "bbox":     [x1, y1, x2, y2],
                    "age":      int,        # frames alive
                    "hits":     int,
                    "face_id":  str | None,
                    "face_name": str | None,
                    "active":   bool
                }
        """
        self.frame_count += 1

        # ── Predict new positions for existing tracks ──────────────────────
        for track in self.tracks:
            track.predict()
            track.time_since_update += 1

        # ── Match detections to tracks ────────────────────────────────────
        matched, unmatched_tracks, unmatched_dets = _greedy_match(
            self.tracks, detections, self.iou_threshold
        )

        # Update matched tracks
        for t_idx, d_idx in matched.items():
            self.tracks[t_idx].update(detections[d_idx]["bbox"])
            self.tracks[t_idx].no_match_count = 0

        # Mark unmatched tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].no_match_count += 1

        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            self.tracks.append(KalmanTrack(detections[d_idx]["bbox"]))

        # ── Prune dead tracks ─────────────────────────────────────────────
        self.tracks = [
            t for t in self.tracks if t.no_match_count <= self.max_age
        ]

        # ── Build output ──────────────────────────────────────────────────
        results = []
        for track in self.tracks:
            is_confirmed = track.hits >= self.min_hits
            results.append(
                {
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "age": int(time.time() - track.created_at),
                    "hits": track.hits,
                    "face_id": track.face_id,
                    "face_name": track.face_name,
                    "face_confidence": track.face_confidence,
                    "active": is_confirmed and track.time_since_update == 0,
                    "confirmed": is_confirmed,
                }
            )

        return results

    def associate_face(
        self,
        track_id: int,
        face_id: Optional[str],
        face_name: Optional[str],
        confidence: float,
    ):
        """Attach a recognised identity to a track."""
        for track in self.tracks:
            if track.track_id == track_id:
                # Only update if new confidence is higher
                if confidence > track.face_confidence:
                    track.face_id = face_id
                    track.face_name = face_name
                    track.face_confidence = confidence
                return

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        KalmanTrack._id_counter = 0

    @property
    def active_track_ids(self) -> List[int]:
        return [t.track_id for t in self.tracks if t.hits >= self.min_hits]
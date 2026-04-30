"""
models/object_tracking.py

Multi-Object Tracking using YOLOv8 built-in tracking (ByteTrack or BoTSORT).

Uses ultralytics model.track() for end-to-end detection + tracking in a single call.
This maintains persistent track IDs across frames automatically.

Responsibilities:
    - Assign persistent track IDs to detected objects across video frames
    - Support ByteTrack and BoTSORT tracker algorithms
    - Provide frame-by-frame tracking updates

Usage:
    tracker = ObjectTrackingModel()
    for frame in video_frames:
        tracks = tracker.update(frame)
    tracker.reset()
"""

import logging
from typing import Dict, List, Optional

import numpy as np

from config.settings import (
    DETECTION_DEVICE,
    DETECTION_MODEL,
    TRACKER_CONFIDENCE,
    TRACKER_IOU,
    TRACKER_TYPE,
)
from utils.helpers import timed

logger = logging.getLogger(__name__)


class ObjectTrackingModel:
    """
    YOLOv8 built-in multi-object tracker using ByteTrack or BoTSORT.

    Unlike traditional SORT implementations, this uses ultralytics' integrated
    model.track() which performs detection and tracking in a single pass.

    Track result schema (per tracked object):
        {
            "track_id":   int,          # persistent ID across frames
            "bbox":       [x1, y1, x2, y2],
            "confidence": float,
            "class":      str           # class name from model
        }

    Args:
        tracker_type: Tracking algorithm — "bytetrack" or "botsort".
        model_path: Path to YOLO weights. Defaults to DETECTION_MODEL.
        conf: Detection confidence threshold for tracking.
        iou: IoU threshold for tracker association.
    """

    def __init__(
        self,
        tracker_type: str = TRACKER_TYPE,
        model_path: Optional[str] = None,
        conf: float = TRACKER_CONFIDENCE,
        iou: float = TRACKER_IOU,
    ):
        self._tracker_type = tracker_type
        self._model_path = model_path or DETECTION_MODEL
        self._conf = conf
        self._iou = iou
        self._model = None
        self._frame_count = 0
        self._load_model()

    def _load_model(self) -> None:
        """Load the YOLOv8 model for tracking."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            logger.info(
                "ObjectTracking: YOLOv8 loaded for tracking (%s, tracker=%s)",
                self._model_path,
                self._tracker_type,
            )
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error("ObjectTracking: model load error - %s", exc)
            raise

    @timed
    def update(self, image: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict]:
        """
        Process a single video frame and return tracked objects.

        Performs detection + tracking in one call using model.track().
        Track IDs persist across sequential calls (stateful).

        Args:
            image: BGR numpy array (H, W, 3) — current video frame.
            classes: Optional list of class IDs to track. None = all.

        Returns:
            List of track dicts with track_id, bbox, confidence, class.
            Returns empty list if no objects are being tracked.
        """
        self._frame_count += 1

        try:
            results = self._model.track(
                image,
                persist=True,
                tracker=f"{self._tracker_type}.yaml",
                conf=self._conf,
                iou=self._iou,
                classes=classes,
                device=DETECTION_DEVICE,
                verbose=False,
            )
        except Exception as exc:
            logger.error("ObjectTracking: track() failed - %s", exc)
            return []

        tracks = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                class_name = self._model.names.get(class_id, str(class_id))

                tracks.append({
                    "track_id": track_id,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class": class_name,
                })

        logger.debug(
            "Frame %d: tracking %d objects", self._frame_count, len(tracks)
        )
        return tracks

    def reset(self) -> None:
        """
        Reset the tracker state.

        Reloads the model to clear all internal tracking state.
        Call this when switching to a new video or scene.
        """
        self._frame_count = 0
        self._load_model()
        logger.info("ObjectTracking: tracker state reset")

    @property
    def frame_count(self) -> int:
        """Number of frames processed since last reset."""
        return self._frame_count

    @property
    def tracker_type(self) -> str:
        """Current tracker algorithm name."""
        return self._tracker_type

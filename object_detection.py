"""
models/object_detection.py

Object Detection Model — YOLOv8 via Ultralytics.

Responsibilities:
  • Detect persons (and optionally other objects) in a frame
  • Return bounding boxes, confidence scores, class labels
  • Support both single-image and batch inference
  • Expose raw detections for downstream tracking / recognition
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    DETECTION_CLASSES,
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
    DETECTION_MODEL,
)
from utils.helpers import timed

logger = logging.getLogger(__name__)

COCO_NAMES = {
    0: "person", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 67: "cell phone",
}


class ObjectDetectionModel:
    """
    YOLOv8-based object detector.

    Detection result schema (per object):
        {
            "bbox":       [x1, y1, x2, y2],   # absolute pixel coords
            "confidence": float,
            "class_id":   int,
            "class_name": str,
            "track_id":   None                 # filled in by tracker
        }
    """

    def __init__(self, classes: Optional[List[int]] = None):
        self.model = None
        self.classes = classes if classes is not None else DETECTION_CLASSES
        self._load_model()

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from ultralytics import YOLO

            self.model = YOLO(DETECTION_MODEL)
            # Warm-up pass
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)
            logger.info("ObjectDetection: YOLOv8 model loaded (%s)", DETECTION_MODEL)

        except ImportError:
            logger.error("Ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error("ObjectDetection: load error — %s", exc)
            raise

    # ─── Inference ────────────────────────────────────────────────────────────

    @timed
    def detect(
        self,
        image: np.ndarray,
        conf: float = DETECTION_CONFIDENCE,
        iou: float = DETECTION_IOU,
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Run detection on a single BGR image.

        Returns a list of detection dicts.
        """
        target_classes = classes if classes is not None else self.classes

        results = self.model(
            image,
            conf=conf,
            iou=iou,
            classes=target_classes if target_classes else None,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = COCO_NAMES.get(class_id, self.model.names.get(class_id, str(class_id)))

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "class_name": class_name,
                        "track_id": None,
                    }
                )

        logger.debug("Detected %d objects", len(detections))
        return detections

    @timed
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """Run detection on a batch of BGR images."""
        batch_results = self.model(images, verbose=False)
        output = []
        for result in batch_results:
            frame_dets = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                class_id = int(box.cls[0])
                frame_dets.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": round(float(box.conf[0]), 4),
                        "class_id": class_id,
                        "class_name": COCO_NAMES.get(class_id, str(class_id)),
                        "track_id": None,
                    }
                )
            output.append(frame_dets)
        return output

    # ─── Person-Specific Helpers ──────────────────────────────────────────────

    def detect_persons(self, image: np.ndarray) -> List[Dict]:
        """Shorthand — detect only persons (class 0)."""
        return self.detect(image, classes=[0])

    # ─── Annotation ───────────────────────────────────────────────────────────

    @staticmethod
    def annotate(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes and labels on a copy of `image`."""
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            if det.get("track_id") is not None:
                label = f"#{det['track_id']} {label}"

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 50), 2)
            cv2.putText(
                out, label, (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 50), 1, cv2.LINE_AA,
            )
        return out

    # ─── Serialisable Output ──────────────────────────────────────────────────

    @staticmethod
    def to_json(detections: List[Dict]) -> List[Dict]:
        """Ensure all values are JSON-serialisable."""
        safe = []
        for d in detections:
            safe.append(
                {
                    "bbox": [int(v) for v in d["bbox"]],
                    "confidence": float(d["confidence"]),
                    "class_id": int(d["class_id"]),
                    "class_name": str(d["class_name"]),
                    "track_id": d.get("track_id"),
                }
            )
        return safe
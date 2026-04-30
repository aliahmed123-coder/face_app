"""
models/object_detection.py

Object Detection using YOLOv8 via the ultralytics package.

Responsibilities:
    - Detect persons, faces, and other objects in a frame
    - Support custom YOLO weights (e.g., yolov8n-face.pt)
    - Return structured detection results with bounding boxes and confidence

Usage:
    detector = ObjectDetectionModel()
    results = detector.detect(image)
    faces = detector.detect_faces(image)
    persons = detector.detect_persons(image)
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    DETECTION_CLASSES,
    DETECTION_CONFIDENCE,
    DETECTION_DEVICE,
    DETECTION_IOU,
    DETECTION_MODEL,
    FACE_DETECTION_MODEL,
)
from utils.helpers import timed

logger = logging.getLogger(__name__)


class ObjectDetectionModel:
    """
    YOLOv8-based object detector supporting both general COCO detection
    and custom face detection weights.

    Detection result schema (per object):
        {
            "bbox":       [x1, y1, x2, y2],   # absolute pixel coordinates
            "confidence": float,               # detection confidence [0, 1]
            "class_id":   int,                 # class index
            "class_name": str                  # human-readable class name
        }

    Args:
        model_path: Path to YOLO weights file. Defaults to DETECTION_MODEL from settings.
        classes: List of class IDs to filter. None = all classes.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        classes: Optional[List[int]] = None,
    ):
        self._model_path = model_path or DETECTION_MODEL
        self._classes = classes if classes is not None else DETECTION_CLASSES
        self._model = None
        self._face_model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the primary YOLOv8 model."""
        try:
            from ultralytics import YOLO

            self._model = YOLO(self._model_path)
            # Warm-up pass to initialise the model
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model(dummy, verbose=False)
            logger.info(
                "ObjectDetection: YOLOv8 model loaded (%s)", self._model_path
            )
        except ImportError:
            logger.error("ultralytics not installed. Run: pip install ultralytics")
            raise
        except Exception as exc:
            logger.error("ObjectDetection: model load error - %s", exc)
            raise

    def _get_face_model(self):
        """Lazy-load the face detection model on first use."""
        if self._face_model is None:
            try:
                from ultralytics import YOLO

                self._face_model = YOLO(FACE_DETECTION_MODEL)
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._face_model(dummy, verbose=False)
                logger.info(
                    "ObjectDetection: face model loaded (%s)", FACE_DETECTION_MODEL
                )
            except Exception as exc:
                logger.warning(
                    "ObjectDetection: face model unavailable (%s), "
                    "falling back to general model",
                    exc,
                )
                self._face_model = self._model
        return self._face_model

    @timed
    def detect(
        self,
        image: np.ndarray,
        conf: float = DETECTION_CONFIDENCE,
        iou: float = DETECTION_IOU,
        classes: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Run object detection on a single BGR image.

        Args:
            image: BGR numpy array (H, W, 3).
            conf: Minimum confidence threshold.
            iou: IoU threshold for NMS.
            classes: Class IDs to detect. None uses instance default.

        Returns:
            List of detection dicts with bbox, confidence, class_id, class_name.
        """
        target_classes = classes if classes is not None else self._classes

        results = self._model(
            image,
            conf=conf,
            iou=iou,
            classes=target_classes if target_classes else None,
            device=DETECTION_DEVICE,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self._model.names.get(class_id, str(class_id))

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                })

        logger.debug("Detected %d objects (conf>=%.2f)", len(detections), conf)
        return detections

    @timed
    def detect_faces(self, image: np.ndarray, conf: float = DETECTION_CONFIDENCE) -> List[Dict]:
        """
        Detect faces using custom face detection weights (yolov8n-face.pt).

        Falls back to general model with person class if face weights unavailable.

        Args:
            image: BGR numpy array.
            conf: Minimum confidence threshold.

        Returns:
            List of detection dicts for faces.
        """
        face_model = self._get_face_model()

        results = face_model(
            image,
            conf=conf,
            device=DETECTION_DEVICE,
            verbose=False,
        )

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = face_model.names.get(class_id, "face")

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(confidence, 4),
                    "class_id": class_id,
                    "class_name": class_name,
                })

        logger.debug("Detected %d faces", len(detections))
        return detections

    @timed
    def detect_persons(self, image: np.ndarray, conf: float = DETECTION_CONFIDENCE) -> List[Dict]:
        """
        Detect only persons (COCO class 0).

        Args:
            image: BGR numpy array.
            conf: Minimum confidence threshold.

        Returns:
            List of detection dicts for persons only.
        """
        return self.detect(image, conf=conf, classes=[0])

    @staticmethod
    def annotate(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on a copy of the image.

        Args:
            image: BGR numpy array.
            detections: List of detection dicts.

        Returns:
            Annotated copy of the image.
        """
        out = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 50), 2)
            cv2.putText(
                out,
                label,
                (x1, max(y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 50),
                1,
                cv2.LINE_AA,
            )
        return out

    @staticmethod
    def to_json(detections: List[Dict]) -> List[Dict]:
        """
        Ensure all detection values are JSON-serialisable.

        Args:
            detections: Raw detection list.

        Returns:
            Cleaned list safe for JSON serialisation.
        """
        return [
            {
                "bbox": [int(v) for v in d["bbox"]],
                "confidence": float(d["confidence"]),
                "class_id": int(d["class_id"]),
                "class_name": str(d["class_name"]),
            }
            for d in detections
        ]

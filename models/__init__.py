"""
models package — Face Attendance System AI models.

Each module is independent and can be imported standalone.
"""

from models.anti_spoofing import AntiSpoofingModel
from models.face_quality import FaceQualityModel
from models.face_recognition import FaceRecognitionModel
from models.fraud_detection import FraudDetectionModel
from models.liveness_detection import LivenessDetectionModel
from models.object_detection import ObjectDetectionModel
from models.object_tracking import ObjectTrackingModel

__all__ = [
    "ObjectDetectionModel",
    "ObjectTrackingModel",
    "AntiSpoofingModel",
    "FaceQualityModel",
    "FaceRecognitionModel",
    "LivenessDetectionModel",
    "FraudDetectionModel",
]

"""
models/face_recognition.py

Face Recognition Model — ArcFace / FaceNet backbone via InsightFace.

Responsibilities:
  • Detect + align faces in an image
  • Extract a 512-d embedding per face
  • Register new identities in the face database
  • Match an embedding against the database
  • Return identity, confidence, and bounding box
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    EMBEDDING_DIM,
    FACE_DB_PATH,
    RECOGNITION_MODEL,
    RECOGNITION_THRESHOLD,
)
from utils.helpers import cosine_similarity, timed

logger = logging.getLogger(__name__)


class FaceRecognitionModel:
    """
    Wraps InsightFace's ArcFace pipeline for end-to-end face recognition.

    Face database schema (pickle):
        {
            "person_id": {
                "name": str,
                "embeddings": List[np.ndarray],   # one or more enrolment shots
                "metadata": dict                  # arbitrary extra fields
            },
            ...
        }
    """

    def __init__(self):
        self.app = None          # InsightFace FaceAnalysis app
        self.face_db: Dict = {}
        self._load_model()
        self._load_db()

    # ─── Initialisation ───────────────────────────────────────────────────────

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis

            self.app = FaceAnalysis(
                name="buffalo_l",                 # ArcFace R100 + RetinaFace
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceRecognition: InsightFace loaded (ArcFace R100)")

        except ImportError:
            logger.warning(
                "InsightFace not installed. Using DeepFace fallback."
            )
            self._use_deepface = True
        except Exception as exc:
            logger.error("FaceRecognition: model load error — %s", exc)
            raise

    def _load_db(self):
        if os.path.exists(FACE_DB_PATH):
            with open(FACE_DB_PATH, "rb") as f:
                self.face_db = pickle.load(f)
            logger.info(
                "FaceRecognition: loaded %d identities from DB", len(self.face_db)
            )

    def _save_db(self):
        with open(FACE_DB_PATH, "wb") as f:
            pickle.dump(self.face_db, f)

    # ─── Core Inference ───────────────────────────────────────────────────────

    @timed
    def detect_and_embed(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in `image` and return per-face dicts:
            {
                "bbox":      [x1, y1, x2, y2],
                "embedding": np.ndarray (512-d, L2-normalised),
                "landmarks": np.ndarray (5, 2),
                "det_score": float
            }
        """
        if self.app is None:
            return self._deepface_detect_embed(image)

        faces = self.app.get(image)
        results = []
        for face in faces:
            emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
            results.append(
                {
                    "bbox": face.bbox.astype(int).tolist(),
                    "embedding": emb,
                    "landmarks": face.kps.astype(int).tolist()
                    if face.kps is not None
                    else [],
                    "det_score": float(face.det_score),
                }
            )
        return results

    def _deepface_fallback_embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Emergency fallback using DeepFace."""
        try:
            from deepface import DeepFace

            result = DeepFace.represent(
                image, model_name="ArcFace", enforce_detection=False
            )
            emb = np.array(result[0]["embedding"])
            return emb / (np.linalg.norm(emb) + 1e-8)
        except Exception as exc:
            logger.error("DeepFace fallback failed: %s", exc)
            return None

    def _deepface_detect_embed(self, image: np.ndarray) -> List[Dict]:
        emb = self._deepface_fallback_embed(image)
        if emb is None:
            return []
        h, w = image.shape[:2]
        return [
            {
                "bbox": [0, 0, w, h],
                "embedding": emb,
                "landmarks": [],
                "det_score": 1.0,
            }
        ]

    # ─── Identification ───────────────────────────────────────────────────────

    @timed
    def identify(self, embedding: np.ndarray) -> Dict:
        """
        Match `embedding` against the face database.

        Returns:
            {
                "person_id":  str | None,
                "name":       str | None,
                "similarity": float,
                "matched":    bool,
                "metadata":   dict
            }
        """
        if not self.face_db:
            return self._no_match(0.0)

        best_sim = -1.0
        best_id = None

        for pid, data in self.face_db.items():
            for ref_emb in data["embeddings"]:
                sim = cosine_similarity(embedding, ref_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = pid

        if best_sim >= RECOGNITION_THRESHOLD:
            person = self.face_db[best_id]
            return {
                "person_id": best_id,
                "name": person["name"],
                "similarity": round(best_sim, 4),
                "matched": True,
                "metadata": person.get("metadata", {}),
            }

        return self._no_match(best_sim)

    @staticmethod
    def _no_match(sim: float) -> Dict:
        return {
            "person_id": None,
            "name": "Unknown",
            "similarity": round(sim, 4),
            "matched": False,
            "metadata": {},
        }

    # ─── Enrolment ────────────────────────────────────────────────────────────

    def enroll(
        self,
        person_id: str,
        name: str,
        image: np.ndarray,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Register a person. Accepts one enrolment image.
        Multiple calls append embeddings (multi-shot enrolment).
        """
        faces = self.detect_and_embed(image)
        if not faces:
            return {"success": False, "reason": "No face detected in enrolment image"}

        # Take the highest-confidence detection
        face = max(faces, key=lambda f: f["det_score"])
        emb = face["embedding"]

        if person_id not in self.face_db:
            self.face_db[person_id] = {
                "name": name,
                "embeddings": [],
                "metadata": metadata or {},
            }

        self.face_db[person_id]["embeddings"].append(emb)
        self.face_db[person_id]["name"] = name  # allow name update
        if metadata:
            self.face_db[person_id]["metadata"].update(metadata)

        self._save_db()
        logger.info(
            "Enrolled %s (%s), total shots: %d",
            name,
            person_id,
            len(self.face_db[person_id]["embeddings"]),
        )
        return {
            "success": True,
            "person_id": person_id,
            "name": name,
            "num_embeddings": len(self.face_db[person_id]["embeddings"]),
        }

    def delete(self, person_id: str) -> bool:
        if person_id in self.face_db:
            del self.face_db[person_id]
            self._save_db()
            return True
        return False

    def list_enrolled(self) -> List[Dict]:
        return [
            {
                "person_id": pid,
                "name": data["name"],
                "num_embeddings": len(data["embeddings"]),
                "metadata": data.get("metadata", {}),
            }
            for pid, data in self.face_db.items()
        ]

    # ─── Full Pipeline ────────────────────────────────────────────────────────

    @timed
    def run(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in `image`, identify each one.

        Returns a list of results (one per detected face):
            {
                "bbox":       [x1, y1, x2, y2],
                "det_score":  float,
                "person_id":  str | None,
                "name":       str,
                "similarity": float,
                "matched":    bool,
                "landmarks":  list
            }
        """
        detections = self.detect_and_embed(image)
        results = []
        for det in detections:
            identity = self.identify(det["embedding"])
            results.append(
                {
                    "bbox": det["bbox"],
                    "det_score": det["det_score"],
                    "landmarks": det["landmarks"],
                    **identity,
                }
            )
        return results
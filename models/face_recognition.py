"""
models/face_recognition.py

Face Recognition using InsightFace (ArcFace R100) with DeepFace fallback.

Responsibilities:
    - Detect and align faces in an image
    - Extract 512-dimensional L2-normalised embeddings
    - Register (enroll) new identities to a pickle-based face database
    - Match embeddings against the database using cosine similarity
    - Manage the face database (CRUD operations)

Usage:
    model = FaceRecognitionModel()
    faces = model.detect_and_embed(image)
    identity = model.identify(faces[0]["embedding"])
    model.enroll("emp_001", "Alice", image)
"""

import logging
import os
import pickle
from typing import Dict, List, Optional

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
    InsightFace-based face recognition with ArcFace R100 backbone.

    Provides end-to-end face detection, embedding extraction, and identity matching.
    Falls back to DeepFace if InsightFace is unavailable.

    Face database schema (pickle file):
        {
            "person_id": {
                "name":       str,
                "embeddings": List[np.ndarray],   # 512-d L2-normalised vectors
                "metadata":   dict                # arbitrary extra fields
            },
            ...
        }

    Embedding properties:
        - Dimension: 512
        - Normalisation: L2 (unit length)
        - Similarity metric: cosine similarity
        - Threshold: configurable via RECOGNITION_THRESHOLD (default 0.60)
    """

    def __init__(self):
        self._app = None
        self._use_deepface: bool = False
        self.face_db: Dict = {}
        self._load_model()
        self._load_db()

    def _load_model(self) -> None:
        """Load InsightFace FaceAnalysis (primary) or flag DeepFace fallback."""
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=RECOGNITION_MODEL,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
            self._use_deepface = False
            logger.info("FaceRecognition: InsightFace loaded (model=%s)", RECOGNITION_MODEL)

        except ImportError:
            logger.warning("InsightFace not installed — using DeepFace fallback")
            self._use_deepface = True
        except Exception as exc:
            logger.error("FaceRecognition: model load error - %s", exc)
            raise

    def _load_db(self) -> None:
        """Load the face database from disk."""
        if os.path.exists(FACE_DB_PATH):
            try:
                with open(FACE_DB_PATH, "rb") as f:
                    self.face_db = pickle.load(f)
                logger.info(
                    "FaceRecognition: loaded %d identities from DB", len(self.face_db)
                )
            except Exception as exc:
                logger.error("FaceRecognition: DB load error - %s", exc)
                self.face_db = {}

    def _save_db(self) -> None:
        """Persist the face database to disk."""
        try:
            os.makedirs(os.path.dirname(FACE_DB_PATH), exist_ok=True)
            with open(FACE_DB_PATH, "wb") as f:
                pickle.dump(self.face_db, f)
        except Exception as exc:
            logger.error("FaceRecognition: DB save error - %s", exc)

    # ─── Core Inference ───────────────────────────────────────────────────────

    @timed
    def detect_and_embed(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in an image and extract embeddings.

        Args:
            image: BGR numpy array.

        Returns:
            List of face dicts:
                {
                    "bbox":      [x1, y1, x2, y2],
                    "embedding": np.ndarray (512-d, L2-normalised),
                    "landmarks": list of [x, y] pairs,
                    "det_score": float
                }
        """
        if self._use_deepface:
            return self._deepface_detect_embed(image)

        faces = self._app.get(image)
        results = []
        for face in faces:
            emb = face.embedding / (np.linalg.norm(face.embedding) + 1e-8)
            results.append({
                "bbox": face.bbox.astype(int).tolist(),
                "embedding": emb,
                "landmarks": face.kps.astype(int).tolist() if face.kps is not None else [],
                "det_score": float(face.det_score),
            })
        return results

    def _deepface_detect_embed(self, image: np.ndarray) -> List[Dict]:
        """Fallback: use DeepFace for detection and embedding."""
        emb = self._deepface_embed(image)
        if emb is None:
            return []
        h, w = image.shape[:2]
        return [{
            "bbox": [0, 0, w, h],
            "embedding": emb,
            "landmarks": [],
            "det_score": 1.0,
        }]

    def _deepface_embed(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding via DeepFace ArcFace."""
        try:
            from deepface import DeepFace

            result = DeepFace.represent(
                image, model_name="ArcFace", enforce_detection=False
            )
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            return emb / (np.linalg.norm(emb) + 1e-8)
        except Exception as exc:
            logger.error("DeepFace fallback failed: %s", exc)
            return None

    # ─── Identification ───────────────────────────────────────────────────────

    @timed
    def identify(self, embedding: np.ndarray) -> Dict:
        """
        Match an embedding against the face database.

        Finds the closest registered identity using cosine similarity.

        Args:
            embedding: 512-d L2-normalised face embedding.

        Returns:
            {
                "person_id":  str | None,
                "name":       str,
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
        """Build a 'not matched' response."""
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
        Register a person's face in the database.

        Accepts one enrolment image per call. Multiple calls for the same
        person_id append embeddings (multi-shot enrolment improves accuracy).

        Args:
            person_id: Unique identifier for the person.
            name: Display name.
            image: BGR image containing the person's face.
            metadata: Optional extra fields (department, employee_id, etc.).

        Returns:
            {
                "success":        bool,
                "person_id":      str,
                "name":           str,
                "num_embeddings": int,
                "reason":         str (only if success=False)
            }
        """
        faces = self.detect_and_embed(image)
        if not faces:
            return {"success": False, "reason": "No face detected in enrolment image"}

        # Use the highest-confidence detection
        face = max(faces, key=lambda f: f["det_score"])
        emb = face["embedding"]

        if person_id not in self.face_db:
            self.face_db[person_id] = {
                "name": name,
                "embeddings": [],
                "metadata": metadata or {},
            }

        self.face_db[person_id]["embeddings"].append(emb)
        self.face_db[person_id]["name"] = name
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
        """
        Remove a person from the face database.

        Args:
            person_id: The identifier to remove.

        Returns:
            True if the person was found and removed, False otherwise.
        """
        if person_id in self.face_db:
            del self.face_db[person_id]
            self._save_db()
            logger.info("Deleted identity: %s", person_id)
            return True
        return False

    def list_enrolled(self) -> List[Dict]:
        """
        List all enrolled identities.

        Returns:
            List of dicts with person_id, name, num_embeddings, metadata.
        """
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
        Full pipeline: detect all faces and identify each one.

        Args:
            image: BGR numpy array.

        Returns:
            List of results per detected face:
                {
                    "bbox":       [x1, y1, x2, y2],
                    "det_score":  float,
                    "landmarks":  list,
                    "person_id":  str | None,
                    "name":       str,
                    "similarity": float,
                    "matched":    bool,
                    "metadata":   dict
                }
        """
        detections = self.detect_and_embed(image)
        results = []
        for det in detections:
            identity = self.identify(det["embedding"])
            results.append({
                "bbox": det["bbox"],
                "det_score": det["det_score"],
                "landmarks": det["landmarks"],
                **identity,
            })
        return results

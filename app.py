"""
api/app.py

Flask REST API — Face Attendance System

Endpoints:
  POST /api/v1/recognize          — Full pipeline: detect + identify + quality + spoof + liveness + fraud
  POST /api/v1/enroll             — Register a new identity
  DELETE /api/v1/enroll/<id>      — Remove an identity
  GET  /api/v1/enrolled           — List all enrolled identities

  POST /api/v1/detect             — Object detection only
  POST /api/v1/track              — Object detection + tracking (video frame)
  POST /api/v1/spoof              — Anti-spoofing check only
  POST /api/v1/quality            — Face quality for each detected face
  POST /api/v1/liveness/session   — Create active liveness challenge session
  POST /api/v1/liveness/verify    — Feed a frame into a liveness session

  GET  /api/v1/fraud/stats/<id>   — Attendance stats for a person
  POST /api/v1/fraud/reset        — Reset fraud detector state (dev only)

  GET  /api/v1/health             — Health check
"""

import logging
import os
import sys
import time
import uuid

from flask import Flask, g, jsonify, request
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import FLASK_DEBUG, FLASK_HOST, FLASK_PORT, MAX_CONTENT_LENGTH
from models.anti_spoofing import AntiSpoofingModel
from models.face_quality import FaceQualityModel
from models.face_recognition import FaceRecognitionModel
from models.fraud_detection import FraudDetectionModel
from models.liveness_detection import LivenessDetectionModel
from models.object_detection import ObjectDetectionModel
from models.object_tracking import ObjectTrackingModel
from utils.helpers import (
    allowed_file,
    decode_image,
    error_response,
    setup_logging,
    success_response,
)

setup_logging(logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    CORS(app)

    # ── Lazy-load models (singleton pattern) ──────────────────────────────────
    _models = {}

    def get_model(name: str):
        if name not in _models:
            logger.info("Loading model: %s", name)
            _models[name] = {
                "recognizer":  lambda: FaceRecognitionModel(),
                "detector":    lambda: ObjectDetectionModel(),
                "tracker":     lambda: ObjectTrackingModel(),
                "spoof":       lambda: AntiSpoofingModel(),
                "liveness":    lambda: LivenessDetectionModel(),
                "fraud":       lambda: FraudDetectionModel(),
                "quality":     lambda: FaceQualityModel(),
            }[name]()
        return _models[name]

    # Convenience aliases used inside routes
    def recognizer()  -> FaceRecognitionModel:   return get_model("recognizer")
    def detector()    -> ObjectDetectionModel:   return get_model("detector")
    def tracker()     -> ObjectTrackingModel:    return get_model("tracker")
    def spoof_model() -> AntiSpoofingModel:      return get_model("spoof")
    def liveness()    -> LivenessDetectionModel: return get_model("liveness")
    def fraud()       -> FraudDetectionModel:    return get_model("fraud")
    def quality()     -> FaceQualityModel:       return get_model("quality")

    # ── Request helpers ───────────────────────────────────────────────────────

    def _load_image():
        """Extract image from multipart/form-data or JSON base64 field."""
        if "image" in request.files:
            f = request.files["image"]
            if not allowed_file(f.filename):
                return None, error_response("File type not allowed", 415)
            image = decode_image(f)
        elif request.is_json and "image" in request.json:
            image = decode_image(request.json["image"])
        else:
            return None, error_response("No image provided (use 'image' field)", 400)

        if image is None:
            return None, error_response("Could not decode image", 422)
        return image, None

    @app.before_request
    def _start_timer():
        g.start_time = time.perf_counter()

    @app.after_request
    def _add_timing(response):
        if hasattr(g, "start_time"):
            elapsed_ms = (time.perf_counter() - g.start_time) * 1000
            response.headers["X-Processing-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH CHECK
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/health", methods=["GET"])
    def health():
        return success_response({"status": "ok", "timestamp": time.time()})

    # ─────────────────────────────────────────────────────────────────────────
    # FULL PIPELINE — /api/v1/recognize
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/recognize", methods=["POST"])
    def recognize():
        """
        Full end-to-end attendance pipeline.

        Form fields (all optional except image):
          image       — face image
          camera_id   — source camera identifier (default "default")
          track_id    — integer track ID from tracking endpoint
          liveness_ok — "true" if liveness was pre-verified
          spoof_ok    — "true" if anti-spoofing was pre-verified
        """
        image, err = _load_image()
        if err:
            return err

        camera_id   = request.form.get("camera_id", "default")
        track_id    = request.form.get("track_id", type=int)
        liveness_ok = request.form.get("liveness_ok", "false").lower() == "true"
        spoof_ok    = request.form.get("spoof_ok",    "false").lower() == "true"

        # 1. Face recognition (detect + embed + identify all faces)
        rec_results = recognizer().run(image)
        if not rec_results:
            return error_response("No face detected in image", 422)

        # 2. Anti-spoofing for each face
        spoof_results = spoof_model().run(image, rec_results)

        # 3. Quality for each face (per-face, independent)
        quality_results = quality().run(image, spoof_results)

        # 4. Liveness (passive, per-face)
        for res in quality_results:
            passive_lv = liveness().passive_check(image, res["bbox"])
            res["liveness"] = passive_lv
            liveness_ok_face = liveness_ok or passive_lv["is_live"]
            spoof_ok_face    = spoof_ok    or res["spoof"]["is_live"]

            # 5. Fraud check per face
            embedding = res.get("embedding")  # may be absent in JSON output
            fraud_result = fraud().check(
                person_id       = res.get("person_id"),
                embedding       = None,          # don't expose raw embedding to fraud module here
                track_id        = track_id,
                bbox            = res["bbox"],
                liveness_passed = liveness_ok_face,
                spoof_passed    = spoof_ok_face,
                camera_id       = camera_id,
            )
            res["fraud"] = fraud_result

        # Strip raw embedding arrays before serialisation
        for res in quality_results:
            res.pop("embedding", None)

        return success_response(
            {
                "camera_id": camera_id,
                "num_faces": len(quality_results),
                "faces": quality_results,
            }
        )

    # ─────────────────────────────────────────────────────────────────────────
    # ENROLMENT
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/enroll", methods=["POST"])
    def enroll():
        """
        Enrol a new or existing identity.

        Form fields:
          image       — enrolment face image
          person_id   — unique identifier (auto-generated UUID if absent)
          name        — display name
          department  — (optional metadata)
          employee_id — (optional metadata)
        """
        image, err = _load_image()
        if err:
            return err

        person_id  = request.form.get("person_id") or str(uuid.uuid4())
        name       = request.form.get("name", "").strip()
        if not name:
            return error_response("'name' field is required", 400)

        metadata = {}
        for key in ("department", "employee_id", "email", "role"):
            val = request.form.get(key)
            if val:
                metadata[key] = val

        result = recognizer().enroll(person_id, name, image, metadata)
        if not result["success"]:
            return error_response(result.get("reason", "Enrollment failed"), 422)

        return success_response(result, status=201)

    @app.route("/api/v1/enroll/<person_id>", methods=["DELETE"])
    def delete_enrolled(person_id: str):
        removed = recognizer().delete(person_id)
        if not removed:
            return error_response(f"Person '{person_id}' not found", 404)
        return success_response({"deleted": person_id})

    @app.route("/api/v1/enrolled", methods=["GET"])
    def list_enrolled():
        people = recognizer().list_enrolled()
        return success_response({"count": len(people), "people": people})

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECT DETECTION
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/detect", methods=["POST"])
    def detect():
        image, err = _load_image()
        if err:
            return err

        conf    = request.form.get("confidence", FLASK_HOST, type=float) or 0.45
        classes = request.form.getlist("classes", type=int) or None
        dets = detector().detect(image, conf=conf, classes=classes)
        return success_response(
            {"num_detections": len(dets), "detections": ObjectDetectionModel.to_json(dets)}
        )

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECT TRACKING (single-frame update)
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/track", methods=["POST"])
    def track():
        image, err = _load_image()
        if err:
            return err

        dets   = detector().detect_persons(image)
        tracks = tracker().update(dets)
        return success_response({"num_tracks": len(tracks), "tracks": tracks})

    @app.route("/api/v1/track/reset", methods=["POST"])
    def track_reset():
        tracker().reset()
        return success_response({"reset": True})

    # ─────────────────────────────────────────────────────────────────────────
    # ANTI-SPOOFING
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/spoof", methods=["POST"])
    def spoof_check():
        image, err = _load_image()
        if err:
            return err

        # First detect faces, then run spoof on each
        rec_results = recognizer().detect_and_embed(image)
        if not rec_results:
            return error_response("No face detected", 422)

        spoof_results = spoof_model().run(image, rec_results)
        # Strip embeddings
        for r in spoof_results:
            r.pop("embedding", None)

        return success_response({"num_faces": len(spoof_results), "faces": spoof_results})

    # ─────────────────────────────────────────────────────────────────────────
    # FACE QUALITY
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/quality", methods=["POST"])
    def quality_check():
        """
        Assess quality for each face in the image independently.
        Returns per-face quality scores and issues.
        """
        image, err = _load_image()
        if err:
            return err

        # Detect faces, then score quality per face
        face_dets = recognizer().detect_and_embed(image)
        if not face_dets:
            return error_response("No face detected", 422)

        # Get head poses for quality scoring
        head_poses = []
        for det in face_dets:
            passive = liveness().passive_check(image, det["bbox"])
            head_poses.append(passive.get("head_pose"))

        quality_results = quality().run(image, face_dets, head_poses=head_poses)

        # Strip raw embeddings
        for r in quality_results:
            r.pop("embedding", None)

        return success_response(
            {"num_faces": len(quality_results), "faces": quality_results}
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIVENESS
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/liveness/session", methods=["POST"])
    def liveness_create_session():
        """
        Create a new challenge-response liveness session.

        JSON body:
          { "challenge": "blink" }   (blink | turn_left | turn_right | nod | smile)
        """
        body      = request.get_json(silent=True) or {}
        challenge = body.get("challenge", "blink")
        session_id = str(uuid.uuid4())
        result = liveness().create_session(session_id, challenge)
        return success_response(result, status=201)

    @app.route("/api/v1/liveness/verify", methods=["POST"])
    def liveness_verify():
        """
        Feed one frame into a liveness session.

        Form fields:
          image      — frame image
          session_id — from /liveness/session response
        """
        image, err = _load_image()
        if err:
            return err

        session_id = request.form.get("session_id")
        if not session_id:
            return error_response("'session_id' is required", 400)

        # Detect a face to get bbox
        faces = recognizer().detect_and_embed(image)
        if not faces:
            return error_response("No face detected in frame", 422)

        bbox   = faces[0]["bbox"]
        result = liveness().update_session(session_id, image, bbox)
        return success_response(result)

    # ─────────────────────────────────────────────────────────────────────────
    # FRAUD
    # ─────────────────────────────────────────────────────────────────────────

    @app.route("/api/v1/fraud/stats/<person_id>", methods=["GET"])
    def fraud_stats(person_id: str):
        stats = fraud().attendance_stats(person_id)
        return success_response(stats)

    @app.route("/api/v1/fraud/reset", methods=["POST"])
    def fraud_reset():
        fraud().reset()
        return success_response({"reset": True})

    # ─────────────────────────────────────────────────────────────────────────
    # ERROR HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    @app.errorhandler(404)
    def not_found(_):
        return error_response("Endpoint not found", 404)

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error_response("Method not allowed", 405)

    @app.errorhandler(413)
    def request_too_large(_):
        return error_response("Image too large (max 16 MB)", 413)

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("Internal error: %s", exc)
        return error_response("Internal server error", 500)

    return app


# ─── Entry Point ──────────────────────────────────────────────────────────────

app = create_app()

if __name__ == "__main__":
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        threaded=True,
    )
"""
models/liveness_detection.py

Liveness Detection — Passive (single-frame) + Active (challenge-response).

Passive analysis (single frame):
    - Head pose estimation (yaw/pitch/roll via solvePnP)
    - Blink detection (Eye Aspect Ratio)
    - Texture-based liveness score

Active analysis (multi-frame session):
    - Challenge-response protocol with session management
    - Supported challenges: blink, turn_left, turn_right, nod, smile
    - State machine: IDLE -> CHALLENGE -> VERIFY -> PASS/FAIL
    - Session timeout and multi-frame verification

Usage:
    model = LivenessDetectionModel()
    # Passive
    result = model.passive_check(image, bbox)
    # Active
    session = model.create_session("sess_123", "blink")
    result = model.update_session("sess_123", frame, bbox)
"""

import logging
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    LIVENESS_BLINK_THRESHOLD,
    LIVENESS_CHALLENGE_FRAMES,
    LIVENESS_HEAD_POSE_THRESHOLD,
    LIVENESS_MIN_FRAMES_PASS,
    LIVENESS_SESSION_TIMEOUT,
)
from utils.helpers import eye_aspect_ratio, timed

logger = logging.getLogger(__name__)


# ─── 3D Face Model Points for Pose Estimation ────────────────────────────────

# Simplified 6-point 3D model (nose tip, chin, eye corners, mouth corners)
_3D_FACE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],         # Nose tip
        [0.0, -330.0, -65.0],    # Chin
        [-225.0, 170.0, -135.0], # Left eye outer corner
        [225.0, 170.0, -135.0],  # Right eye outer corner
        [-150.0, -150.0, -125.0],# Left mouth corner
        [150.0, -150.0, -125.0], # Right mouth corner
    ],
    dtype=np.float64,
)


# ─── Head Pose Estimation ─────────────────────────────────────────────────────


def estimate_head_pose(
    landmarks_2d: np.ndarray, image_size: Tuple[int, int]
) -> Dict[str, float]:
    """
    Estimate head orientation (yaw, pitch, roll) from 2D landmarks using solvePnP.

    Args:
        landmarks_2d: Array of shape (6, 2) matching _3D_FACE_POINTS.
        image_size: Tuple (width, height) of the image.

    Returns:
        Dict with "pitch", "yaw", "roll" in degrees.
    """
    w, h = image_size
    focal_length = w
    cam_matrix = np.array(
        [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, _ = cv2.solvePnP(
        _3D_FACE_POINTS,
        landmarks_2d.astype(np.float64),
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    rot_mat, _ = cv2.Rodrigues(rot_vec)

    # Decompose rotation matrix to Euler angles
    sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = np.arctan2(rot_mat[2, 1], rot_mat[2, 2])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
    else:
        x = np.arctan2(-rot_mat[1, 2], rot_mat[1, 1])
        y = np.arctan2(-rot_mat[2, 0], sy)
        z = 0.0

    return {
        "pitch": round(float(np.degrees(x)), 2),
        "yaw": round(float(np.degrees(y)), 2),
        "roll": round(float(np.degrees(z)), 2),
    }


# ─── Challenge Types ──────────────────────────────────────────────────────────


class ChallengeType(Enum):
    """Supported active liveness challenges."""

    BLINK = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    NOD = auto()
    SMILE = auto()


# ─── Liveness Session ─────────────────────────────────────────────────────────


class LivenessSession:
    """
    Stateful session for active liveness verification.

    Tracks frames processed, challenge progress, and session lifetime.
    """

    def __init__(self, session_id: str, challenge: ChallengeType):
        self.session_id = session_id
        self.challenge = challenge
        self.state = "IDLE"
        self.frames_processed = 0
        self.challenge_met = False
        self.created_at = time.time()
        self.blink_count = 0
        self.ear_history: List[float] = []
        self.pose_history: List[Dict] = []

    @property
    def expired(self) -> bool:
        """Whether the session has timed out."""
        return (time.time() - self.created_at) > LIVENESS_SESSION_TIMEOUT

    @property
    def passed(self) -> bool:
        """Whether the challenge was met with enough frames."""
        return self.challenge_met and self.frames_processed >= LIVENESS_MIN_FRAMES_PASS

    @property
    def failed(self) -> bool:
        """Whether the session used all frames without meeting the challenge."""
        return (not self.challenge_met) and (
            self.frames_processed >= LIVENESS_CHALLENGE_FRAMES
        )


# ─── Main Liveness Detection Model ───────────────────────────────────────────


class LivenessDetectionModel:
    """
    Combined passive + active liveness detection.

    Passive mode: single-frame analysis (head pose, blink, texture).
    Active mode: multi-frame challenge-response sessions.

    Passive result schema:
        {
            "ear":                  float,
            "blink":                bool,
            "head_pose":            {"pitch": float, "yaw": float, "roll": float},
            "frontal":              bool,
            "texture_liveness":     float,
            "score":                float,   # composite score [0, 1]
            "is_live":              bool
        }
    """

    def __init__(self):
        self._sessions: Dict[str, LivenessSession] = {}
        self._face_mesh = None
        self._mp_face_mesh = None
        self._load_face_mesh()
        logger.info("LivenessDetection: model initialised")

    def _load_face_mesh(self) -> None:
        """Load MediaPipe FaceMesh for landmark extraction."""
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            logger.info("LivenessDetection: MediaPipe FaceMesh loaded")
        except ImportError:
            logger.warning(
                "MediaPipe not installed - liveness will use reduced accuracy mode"
            )

    # ─── Passive Analysis ─────────────────────────────────────────────────────

    @timed
    def passive_check(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Analyse a single face for passive liveness cues.

        Computes eye aspect ratio (blink), head pose, and texture liveness
        without requiring user interaction.

        Args:
            image: Full BGR image.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Passive liveness result dict.
        """
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return self._empty_passive()

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        ear = 0.3  # Default EAR (open eyes)
        head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        texture_score = 0.5

        if self._face_mesh is not None:
            results = self._face_mesh.process(face_rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                h_f, w_f = face.shape[:2]

                # EAR from MediaPipe eye landmarks
                left_eye = np.array([
                    [lms[33].x * w_f, lms[33].y * h_f],
                    [lms[160].x * w_f, lms[160].y * h_f],
                    [lms[158].x * w_f, lms[158].y * h_f],
                    [lms[133].x * w_f, lms[133].y * h_f],
                    [lms[153].x * w_f, lms[153].y * h_f],
                    [lms[144].x * w_f, lms[144].y * h_f],
                ])
                right_eye = np.array([
                    [lms[362].x * w_f, lms[362].y * h_f],
                    [lms[385].x * w_f, lms[385].y * h_f],
                    [lms[387].x * w_f, lms[387].y * h_f],
                    [lms[263].x * w_f, lms[263].y * h_f],
                    [lms[373].x * w_f, lms[373].y * h_f],
                    [lms[380].x * w_f, lms[380].y * h_f],
                ])
                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                # Head pose from 6 landmarks
                pts_2d = np.array([
                    [lms[4].x * w_f, lms[4].y * h_f],       # Nose tip
                    [lms[152].x * w_f, lms[152].y * h_f],   # Chin
                    [lms[226].x * w_f, lms[226].y * h_f],   # Left eye outer
                    [lms[446].x * w_f, lms[446].y * h_f],   # Right eye outer
                    [lms[57].x * w_f, lms[57].y * h_f],     # Left mouth corner
                    [lms[287].x * w_f, lms[287].y * h_f],   # Right mouth corner
                ])
                head_pose = estimate_head_pose(pts_2d, (w_f, h_f))

        # Texture liveness score (LBP entropy-based)
        texture_score = self._texture_liveness(face)

        blink = ear < LIVENESS_BLINK_THRESHOLD
        thr = LIVENESS_HEAD_POSE_THRESHOLD
        frontal = abs(head_pose["yaw"]) < thr and abs(head_pose["pitch"]) < thr

        # Composite score
        ear_score = min(ear / 0.35, 1.0)
        pose_penalty = max(
            0.0,
            1.0 - max(abs(head_pose["yaw"]) / 90.0, abs(head_pose["pitch"]) / 90.0),
        )
        score = round(0.4 * ear_score + 0.3 * pose_penalty + 0.3 * texture_score, 4)

        return {
            "ear": round(ear, 4),
            "blink": blink,
            "head_pose": head_pose,
            "frontal": frontal,
            "texture_liveness": round(texture_score, 4),
            "score": score,
            "is_live": score >= 0.4 and frontal,
        }

    @staticmethod
    def _texture_liveness(face_bgr: np.ndarray) -> float:
        """Quick texture liveness score via Laplacian energy."""
        gray = cv2.cvtColor(cv2.resize(face_bgr, (64, 64)), cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        energy = float(np.mean(np.abs(lap)))
        return float(np.clip(energy / 30.0, 0.0, 1.0))

    @staticmethod
    def _empty_passive() -> Dict:
        """Return an empty/failed passive result."""
        return {
            "ear": 0.0,
            "blink": False,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "frontal": False,
            "texture_liveness": 0.0,
            "score": 0.0,
            "is_live": False,
        }

    # ─── Active / Session-Based ───────────────────────────────────────────────

    def create_session(self, session_id: str, challenge: str = "blink") -> Dict:
        """
        Create a new challenge-response liveness session.

        Args:
            session_id: Unique session identifier.
            challenge: Challenge type — "blink", "turn_left", "turn_right", "nod", "smile".

        Returns:
            Session info dict with session_id, challenge, instruction, max_frames.
        """
        challenge_map = {
            "blink": ChallengeType.BLINK,
            "turn_left": ChallengeType.TURN_LEFT,
            "turn_right": ChallengeType.TURN_RIGHT,
            "nod": ChallengeType.NOD,
            "smile": ChallengeType.SMILE,
        }
        ctype = challenge_map.get(challenge, ChallengeType.BLINK)
        session = LivenessSession(session_id, ctype)
        session.state = "CHALLENGE"
        self._sessions[session_id] = session

        return {
            "session_id": session_id,
            "challenge": challenge,
            "instruction": self._challenge_instruction(ctype),
            "max_frames": LIVENESS_CHALLENGE_FRAMES,
            "timeout_seconds": LIVENESS_SESSION_TIMEOUT,
        }

    def update_session(self, session_id: str, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Feed a video frame into an active liveness session.

        Analyses the frame for the required challenge action and updates session state.

        Args:
            session_id: Active session identifier.
            image: BGR video frame.
            bbox: Face bounding box [x1, y1, x2, y2].

        Returns:
            Session status dict with progress information.
            Includes "error" key if session is invalid or expired.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return {"error": "Session not found"}
        if session.expired:
            del self._sessions[session_id]
            return {"error": "Session expired", "passed": False}

        passive = self.passive_check(image, bbox)
        session.ear_history.append(passive["ear"])
        session.pose_history.append(passive["head_pose"])
        session.frames_processed += 1

        # Evaluate the specific challenge
        self._evaluate_challenge(session, passive)

        result = {
            "session_id": session_id,
            "frames_processed": session.frames_processed,
            "challenge": session.challenge.name.lower(),
            "challenge_met": session.challenge_met,
            "passed": session.passed,
            "failed": session.failed,
            "passive": passive,
        }

        # Clean up completed sessions
        if session.passed or session.failed:
            del self._sessions[session_id]

        return result

    def _evaluate_challenge(self, session: LivenessSession, passive: Dict) -> None:
        """Evaluate whether the current frame satisfies the active challenge."""
        if session.challenge == ChallengeType.BLINK:
            # Detect a blink transition (open -> closed)
            if passive["blink"] and len(session.ear_history) >= 2:
                if session.ear_history[-2] >= LIVENESS_BLINK_THRESHOLD:
                    session.blink_count += 1
            session.challenge_met = session.blink_count >= 1

        elif session.challenge in (ChallengeType.TURN_LEFT, ChallengeType.TURN_RIGHT):
            thr = LIVENESS_HEAD_POSE_THRESHOLD * 2
            yaws = [p["yaw"] for p in session.pose_history]
            if session.challenge == ChallengeType.TURN_LEFT:
                session.challenge_met = any(y < -thr for y in yaws)
            else:
                session.challenge_met = any(y > thr for y in yaws)

        elif session.challenge == ChallengeType.NOD:
            pitches = [p["pitch"] for p in session.pose_history]
            if len(pitches) >= 2:
                session.challenge_met = (max(pitches) - min(pitches)) > 15.0

        elif session.challenge == ChallengeType.SMILE:
            # Smile detection via EAR proxy (smiling narrows eyes slightly)
            # This is a simplified heuristic
            if len(session.ear_history) >= 3:
                baseline = np.mean(session.ear_history[:3])
                recent = np.mean(session.ear_history[-3:])
                session.challenge_met = (baseline - recent) > 0.02

    @staticmethod
    def _challenge_instruction(challenge: ChallengeType) -> str:
        """Get the user-facing instruction for a challenge type."""
        instructions = {
            ChallengeType.BLINK: "Please blink your eyes",
            ChallengeType.TURN_LEFT: "Please turn your head to the left",
            ChallengeType.TURN_RIGHT: "Please turn your head to the right",
            ChallengeType.NOD: "Please nod your head up and down",
            ChallengeType.SMILE: "Please smile",
        }
        return instructions[challenge]

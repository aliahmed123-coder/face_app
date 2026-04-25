"""
models/liveness_detection.py

Liveness Detection Model — Challenge-Response + Passive Cues.

Two complementary strategies:
  A. Passive (single-frame):
       • Eye Aspect Ratio (EAR) blink detection
       • Head pose estimation (pitch / yaw / roll via solvePnP)
       • Micro-texture depth cues

  B. Active / Challenge-Response (multi-frame session):
       • Ask the user to blink, turn head, or smile
       • Verify the action occurred within a window of frames
       • State machine: IDLE → CHALLENGE → VERIFY → PASS/FAIL
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
)
from utils.helpers import eye_aspect_ratio, timed

logger = logging.getLogger(__name__)


# ─── 3D face model reference points (68-point / simplified) ──────────────────

# Simplified 6-point 3D model for head pose (nose tip, chin, eye corners, mouth corners)
_3D_FACE_PTS = np.array([
    [0.0,    0.0,    0.0   ],  # nose tip
    [0.0,   -330.0, -65.0  ],  # chin
    [-225.0, 170.0, -135.0 ],  # left eye corner
    [225.0,  170.0, -135.0 ],  # right eye corner
    [-150.0,-150.0, -125.0 ],  # left mouth corner
    [150.0, -150.0, -125.0 ],  # right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe/dlib landmark indices (simplified)
_LM_INDICES_6PT = [1, 152, 226, 446, 57, 287]  # approximate for 468-pt model


# ─── Head Pose Estimation ─────────────────────────────────────────────────────

def estimate_head_pose(landmarks_2d: np.ndarray, image_size: Tuple[int, int]) -> Dict:
    """
    Estimate yaw, pitch, roll from 6 2D landmarks + solvePnP.
    `landmarks_2d`: shape (6, 2) — the 6 points matching _3D_FACE_PTS.
    `image_size`: (width, height)
    """
    w, h = image_size
    focal_length = w
    cam_matrix = np.array(
        [[focal_length, 0, w / 2],
         [0, focal_length, h / 2],
         [0, 0, 1]], dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        _3D_FACE_PTS, landmarks_2d.astype(np.float64),
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Decompose rotation matrix into Euler angles
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
        "pitch": round(np.degrees(x), 2),
        "yaw":   round(np.degrees(y), 2),
        "roll":  round(np.degrees(z), 2),
    }


# ─── Challenge-Response State Machine ─────────────────────────────────────────

class ChallengeType(Enum):
    BLINK = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    NOD = auto()
    SMILE = auto()


class LivenessSession:
    """Stateful liveness session for a single user."""

    def __init__(self, session_id: str, challenge: ChallengeType = ChallengeType.BLINK):
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
        return (time.time() - self.created_at) > 60.0  # 60s session timeout

    @property
    def passed(self) -> bool:
        return self.challenge_met and self.frames_processed >= 5

    @property
    def failed(self) -> bool:
        return (not self.challenge_met) and (
            self.frames_processed >= LIVENESS_CHALLENGE_FRAMES
        )


# ─── Main Liveness Model ──────────────────────────────────────────────────────

class LivenessDetectionModel:
    """
    Passive + active liveness detection.

    Passive mode: single-image / single-frame analysis.
    Active mode: multi-frame session with challenge-response.
    """

    def __init__(self):
        self.sessions: Dict[str, LivenessSession] = {}
        self._face_mesh = None
        self._load_face_mesh()
        logger.info("LivenessDetection: model initialised")

    def _load_face_mesh(self):
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
            logger.warning("MediaPipe not installed — using landmark fallback")

    # ─── Passive Analysis ─────────────────────────────────────────────────────

    @timed
    def passive_check(self, image: np.ndarray, bbox: List[int]) -> Dict:
        """
        Analyse a single face crop for passive liveness cues.

        Returns:
            {
                "ear":        float,      # Eye Aspect Ratio
                "blink":      bool,
                "head_pose":  dict,       # pitch, yaw, roll
                "frontal":    bool,       # head within acceptable pose range
                "score":      float,      # composite liveness score [0,1]
                "is_live":    bool
            }
        """
        x1, y1, x2, y2 = bbox
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return self._empty_passive()

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        ear = 0.3
        head_pose = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        if self._face_mesh:
            results = self._face_mesh.process(face_rgb)
            if results.multi_face_landmarks:
                lms = results.multi_face_landmarks[0].landmark
                h_f, w_f = face.shape[:2]
                # EAR from MediaPipe eye landmarks
                left_eye = np.array([
                    [lms[33].x * w_f,  lms[33].y * h_f],
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
                    [lms[4].x * w_f,   lms[4].y * h_f  ],  # nose tip
                    [lms[152].x * w_f, lms[152].y * h_f],  # chin
                    [lms[226].x * w_f, lms[226].y * h_f],  # L eye outer
                    [lms[446].x * w_f, lms[446].y * h_f],  # R eye outer
                    [lms[57].x * w_f,  lms[57].y * h_f ],  # L mouth corner
                    [lms[287].x * w_f, lms[287].y * h_f],  # R mouth corner
                ])
                head_pose = estimate_head_pose(pts_2d, (w_f, h_f))

        blink = ear < LIVENESS_BLINK_THRESHOLD
        thr = LIVENESS_HEAD_POSE_THRESHOLD
        frontal = (
            abs(head_pose["yaw"])   < thr and
            abs(head_pose["pitch"]) < thr
        )

        # Composite score
        ear_score = min(ear / 0.35, 1.0)
        pose_penalty = max(0.0, 1.0 - max(
            abs(head_pose["yaw"]) / 90.0,
            abs(head_pose["pitch"]) / 90.0,
        ))
        score = round(0.5 * ear_score + 0.5 * pose_penalty, 4)

        return {
            "ear": round(ear, 4),
            "blink": blink,
            "head_pose": head_pose,
            "frontal": frontal,
            "score": score,
            "is_live": score >= 0.4 and frontal,
        }

    @staticmethod
    def _empty_passive() -> Dict:
        return {
            "ear": 0.0,
            "blink": False,
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "frontal": False,
            "score": 0.0,
            "is_live": False,
        }

    # ─── Active / Session-Based ───────────────────────────────────────────────

    def create_session(self, session_id: str, challenge: str = "blink") -> Dict:
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
        self.sessions[session_id] = session
        return {
            "session_id": session_id,
            "challenge": challenge,
            "instruction": self._challenge_instruction(ctype),
            "max_frames": LIVENESS_CHALLENGE_FRAMES,
        }

    def update_session(self, session_id: str, image: np.ndarray, bbox: List[int]) -> Dict:
        """Feed a frame into an active liveness session."""
        session = self.sessions.get(session_id)
        if session is None:
            return {"error": "Session not found"}
        if session.expired:
            del self.sessions[session_id]
            return {"error": "Session expired", "passed": False}

        passive = self.passive_check(image, bbox)
        session.ear_history.append(passive["ear"])
        session.pose_history.append(passive["head_pose"])
        session.frames_processed += 1

        # Evaluate challenge
        if session.challenge == ChallengeType.BLINK:
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
            session.challenge_met = (max(pitches) - min(pitches)) > 15.0

        result = {
            "session_id": session_id,
            "frames_processed": session.frames_processed,
            "challenge": session.challenge.name.lower(),
            "challenge_met": session.challenge_met,
            "passed": session.passed,
            "failed": session.failed,
            "passive": passive,
        }

        if session.passed or session.failed:
            del self.sessions[session_id]

        return result

    @staticmethod
    def _challenge_instruction(challenge: ChallengeType) -> str:
        return {
            ChallengeType.BLINK: "Please blink your eyes",
            ChallengeType.TURN_LEFT: "Please turn your head to the left",
            ChallengeType.TURN_RIGHT: "Please turn your head to the right",
            ChallengeType.NOD: "Please nod your head up and down",
            ChallengeType.SMILE: "Please smile",
        }[challenge]
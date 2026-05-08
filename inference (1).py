# ============================================================
#  inference.py  —  Real-time Fatigue Detection Engine
#
#  يشتغل على:
#      - Webcam frame by frame
#      - Video file frame by frame
#
#  Pipeline لكل frame:
#      1. MediaPipe Face Mesh → 468 landmarks
#      2. ROI Extraction      → Eye crop + Mouth crop
#      3. EAR / MAR / Head Pose calculation
#      4. CNN Model inference
#      5. Temporal Smoothing
#      6. Fatigue Score → ALERT / WARNING / DANGER
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import math

from config import *


# ──────────────────────────────────────────────────────────────
# MEDIAPIPE SETUP
# ──────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils


# ──────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────
def _euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def compute_ear(landmarks, eye_indices, frame_w, frame_h):
    """
    EAR — Eye Aspect Ratio
    ─────────────────────
    Formula:  EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)

    - p1, p4  → horizontal corners (العرض)
    - p2,p3,p5,p6 → vertical points (الارتفاع)

    لما العين مفتوحة  → EAR ≈ 0.3+
    لما العين مغلقة  → EAR ≈ 0.15-
    Threshold ≈ 0.25
    """
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
           for i in eye_indices]

    # vert1 = ||p2-p6||,  vert2 = ||p3-p5||
    vert1 = _euclidean(pts[1], pts[5])
    vert2 = _euclidean(pts[2], pts[4])
    horiz = _euclidean(pts[0], pts[3])

    ear = (vert1 + vert2) / (2.0 * horiz + 1e-6)
    return ear


def compute_mar(landmarks, mouth_indices, frame_w, frame_h):
    """
    MAR — Mouth Aspect Ratio
    ────────────────────────
    مشابه للـ EAR بس للفم:
    MAR = vertical_distance / horizontal_distance

    لما الفم مفتوح (تثاؤب) → MAR > 0.6
    """
    pts = [(int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
           for i in mouth_indices]

    vert  = _euclidean(pts[2], pts[6])   # فوق لتحت
    horiz = _euclidean(pts[0], pts[4])   # يمين لشمال

    mar = vert / (horiz + 1e-6)
    return mar


def compute_head_pose_pitch(landmarks, frame_w, frame_h):
    """
    Head Pose Estimation (Pitch فقط — الميل للأمام/الخلف)
    ────────────────────────────────────────────────────
    بنستخدم 6 نقاط من الوجه مع solvePnP:
        - Nose tip
        - Chin
        - Left/Right eye corner
        - Left/Right mouth corner

    solvePnP بيحسب الـ rotation vector من 3D model points
    وبيحولها لزوايا (pitch, yaw, roll)

    Pitch > 20 درجة → رأس بايظ للأمام = drowsy
    """
    # 3D نقاط الوجه في الـ real world (model points)
    model_points = np.array([
        [0.0,    0.0,    0.0  ],   # Nose tip
        [0.0,   -63.6, -12.5 ],   # Chin
        [-43.3,  32.7, -26.0 ],   # Left eye corner
        [43.3,   32.7, -26.0 ],   # Right eye corner
        [-28.9, -28.9, -24.1 ],   # Left mouth corner
        [28.9,  -28.9, -24.1 ],   # Right mouth corner
    ], dtype=np.float64)

    # 2D نقاط مقابلة من MediaPipe
    landmark_ids = [1, 152, 263, 33, 287, 57]
    image_points = np.array([
        (landmarks[i].x * frame_w, landmarks[i].y * frame_h)
        for i in landmark_ids
    ], dtype=np.float64)

    # Camera intrinsics (تقريبية)
    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    cam_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points,
        cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0

    # Rotation vector → Rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    pitch = angles[0]    # الميل للأمام/الخلف
    return pitch


# ──────────────────────────────────────────────────────────────
# ROI EXTRACTION
# ──────────────────────────────────────────────────────────────
def extract_eye_roi(frame, landmarks, eye_indices, frame_w, frame_h,
                    padding=20, target_size=IMG_SIZE):
    """
    Region of Interest Extraction — العين
    ──────────────────────────────────────
    Steps:
        1. خذ الـ landmark points بتاعة العين
        2. احسب الـ bounding box حواليها
        3. أضف padding عشان ما تقطعش الحواف
        4. Crop الـ region دي من الـ frame
        5. Resize للـ model input size

    ده الـ segmentation بتاعنا — مش pixel-level segmentation
    ده spatial segmentation = عزل منطقة معينة من الصورة
    """
    pts = np.array([
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in eye_indices
    ])

    x_min = max(0,        pts[:, 0].min() - padding)
    x_max = min(frame_w,  pts[:, 0].max() + padding)
    y_min = max(0,        pts[:, 1].min() - padding)
    y_max = min(frame_h,  pts[:, 1].max() + padding)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, target_size)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi


def extract_mouth_roi(frame, landmarks, mouth_indices, frame_w, frame_h,
                      padding=25, target_size=IMG_SIZE):
    """نفس extract_eye_roi بس للفم — padding أكبر شوية"""
    pts = np.array([
        (int(landmarks[i].x * frame_w), int(landmarks[i].y * frame_h))
        for i in mouth_indices
    ])

    x_min = max(0,        pts[:, 0].min() - padding)
    x_max = min(frame_w,  pts[:, 0].max() + padding)
    y_min = max(0,        pts[:, 1].min() - padding)
    y_max = min(frame_h,  pts[:, 1].max() + padding)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, target_size)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi


def extract_full_face_roi(frame, landmarks, frame_w, frame_h,
                           padding=30, target_size=IMG_SIZE):
    """Crop الوجه كله للـ Head Nodding model"""
    all_pts = np.array([
        (int(lm.x * frame_w), int(lm.y * frame_h))
        for lm in landmarks
    ])

    x_min = max(0,        all_pts[:, 0].min() - padding)
    x_max = min(frame_w,  all_pts[:, 0].max() + padding)
    y_min = max(0,        all_pts[:, 1].min() - padding)
    y_max = min(frame_h,  all_pts[:, 1].max() + padding)

    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, target_size)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return roi


# ──────────────────────────────────────────────────────────────
# PREPROCESS FOR CNN
# ──────────────────────────────────────────────────────────────
def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Normalize + add batch dimension"""
    img = roi.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)   # (1, H, W, 3)


# ──────────────────────────────────────────────────────────────
# FATIGUE ENGINE
# ──────────────────────────────────────────────────────────────
class FatigueEngine:
    """
    الـ Engine الرئيسي — يجمع كل الـ models في pipeline واحد

    لكل frame:
        → MediaPipe → ROI extraction → CNN inference
        → EAR/MAR/Pitch calculation
        → Score fusion → Temporal smoothing
        → Output: fatigue level + annotations
    """

    def __init__(self, eye_model_path: str, mouth_model_path: str,
                 head_model_path: str, microsleep_model_path: str = None):

        print("Loading models...")
        self.eye_model   = tf.keras.models.load_model(eye_model_path)
        self.mouth_model = tf.keras.models.load_model(mouth_model_path)
        self.head_model  = tf.keras.models.load_model(head_model_path)
        self.micro_model = (tf.keras.models.load_model(microsleep_model_path)
                            if microsleep_model_path else None)

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_CONFIDENCE,
            min_tracking_confidence=LANDMARK_CONFIDENCE
        )

        # Temporal smoothing buffers
        self.eye_scores    = deque(maxlen=TEMPORAL_WINDOW)
        self.mouth_scores  = deque(maxlen=TEMPORAL_WINDOW)
        self.head_scores   = deque(maxlen=TEMPORAL_WINDOW)
        self.ear_history   = deque(maxlen=TEMPORAL_WINDOW)
        self.frame_buffer  = deque(maxlen=MICROSLEEP_FRAMES)  # for LSTM

        # Session tracking
        self.fatigue_log   = []    # [(timestamp, level, score)]
        self.frame_count   = 0

    # ── Process single frame ───────────────────────────────
    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Input:  BGR frame من OpenCV
        Output: dict مع كل النتايج والـ annotated frame
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        result = {
            "frame": frame.copy(),
            "face_detected": False,
            "ear": 0.0,
            "mar": 0.0,
            "pitch": 0.0,
            "eye_prob": 0.0,       # prob إن العين مغلقة
            "mouth_prob": 0.0,     # prob إن في تثاؤب
            "head_prob": 0.0,      # prob إن الرأس بايظ
            "microsleep_prob": 0.0,
            "fatigue_score": 0.0,  # 0-100
            "fatigue_level": "ALERT",
            "fatigue_type": [],    # أنواع الإرهاق الموجودة
        }

        # Convert BGR → RGB للـ MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = self.face_mesh.process(rgb)

        if not mesh_result.multi_face_landmarks:
            return result   # مفيش وجه

        result["face_detected"] = True
        landmarks = mesh_result.multi_face_landmarks[0].landmark

        # ── 1. EAR & MAR (geometric) ──────────────────────
        left_ear  = compute_ear(landmarks, LEFT_EYE_IDX,  w, h)
        right_ear = compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
        ear = (left_ear + right_ear) / 2.0
        mar = compute_mar(landmarks, MOUTH_IDX, w, h)
        pitch = compute_head_pose_pitch(landmarks, w, h)

        result["ear"]   = ear
        result["mar"]   = mar
        result["pitch"] = pitch
        self.ear_history.append(ear)

        # ── 2. ROI Extraction ─────────────────────────────
        left_eye_roi  = extract_eye_roi(frame, landmarks, LEFT_EYE_IDX,  w, h)
        right_eye_roi = extract_eye_roi(frame, landmarks, RIGHT_EYE_IDX, w, h)
        mouth_roi     = extract_mouth_roi(frame, landmarks, MOUTH_IDX, w, h)
        face_roi      = extract_full_face_roi(frame, landmarks, w, h)

        # ── 3. CNN Inference ──────────────────────────────
        # Eye
        if left_eye_roi is not None and right_eye_roi is not None:
            left_prob  = float(self.eye_model.predict(preprocess_roi(left_eye_roi),  verbose=0)[0][0])
            right_prob = float(self.eye_model.predict(preprocess_roi(right_eye_roi), verbose=0)[0][0])
            eye_prob = (left_prob + right_prob) / 2.0   # average العينين
        else:
            eye_prob = float(ear < EAR_THRESHOLD)       # fallback للـ EAR
        result["eye_prob"] = eye_prob
        self.eye_scores.append(eye_prob)

        # Mouth
        if mouth_roi is not None:
            mouth_prob = float(self.mouth_model.predict(preprocess_roi(mouth_roi), verbose=0)[0][0])
        else:
            mouth_prob = float(mar > MAR_THRESHOLD)
        result["mouth_prob"] = mouth_prob
        self.mouth_scores.append(mouth_prob)

        # Head
        if face_roi is not None:
            head_prob = float(self.head_model.predict(preprocess_roi(face_roi), verbose=0)[0][0])
        else:
            head_prob = float(abs(pitch) > PITCH_THRESHOLD)
        result["head_prob"] = head_prob
        self.head_scores.append(head_prob)

        # ── 4. PERCLOS (% of time eyes closed in window) ──
        perclos = (np.array(self.ear_history) < EAR_THRESHOLD).mean()

        # ── 5. Microsleep (CNN + LSTM) ────────────────────
        if face_roi is not None:
            self.frame_buffer.append(face_roi.astype(np.float32) / 255.0)

        microsleep_prob = 0.0
        if self.micro_model and len(self.frame_buffer) == MICROSLEEP_FRAMES:
            seq = np.expand_dims(np.array(self.frame_buffer), axis=0)   # (1, 30, H, W, 3)
            microsleep_prob = float(self.micro_model.predict(seq, verbose=0)[0][0])
        result["microsleep_prob"] = microsleep_prob

        # ── 6. Score Fusion ───────────────────────────────
        # Weighted average مع temporal smoothing
        smooth_eye   = np.mean(self.eye_scores)   if self.eye_scores   else eye_prob
        smooth_mouth = np.mean(self.mouth_scores) if self.mouth_scores else mouth_prob
        smooth_head  = np.mean(self.head_scores)  if self.head_scores  else head_prob

        # Weights: Eye أهم من Mouth, Microsleep أخطر حاجة
        fatigue_score = (
            smooth_eye       * 30 +   # 30%
            smooth_mouth     * 15 +   # 15%
            smooth_head      * 20 +   # 20%
            perclos          * 25 +   # 25%
            microsleep_prob  * 10     # 10%
        )
        result["fatigue_score"] = fatigue_score

        # ── 7. Fatigue Level Classification ───────────────
        if fatigue_score < 30:
            level = "ALERT"
        elif fatigue_score < 60:
            level = "WARNING"
        else:
            level = "DANGER"
        result["fatigue_level"] = level

        # Fatigue types
        types = []
        if smooth_eye   > 0.5: types.append("Eye Fatigue")
        if smooth_mouth > 0.5: types.append("Yawning")
        if smooth_head  > 0.5: types.append("Head Nodding")
        if microsleep_prob > 0.5: types.append("Microsleep")
        result["fatigue_type"] = types

        # ── 8. Draw Annotations ───────────────────────────
        result["frame"] = self._annotate(frame, landmarks, result, w, h)

        return result

    # ── Annotation ────────────────────────────────────────
    def _annotate(self, frame, landmarks, result, w, h):
        ann = frame.copy()

        color_map = {"ALERT": (0, 200, 0), "WARNING": (0, 200, 255), "DANGER": (0, 0, 255)}
        color = color_map[result["fatigue_level"]]

        # Score bar
        bar_w = int(result["fatigue_score"] / 100 * 250)
        cv2.rectangle(ann, (10, 10), (260, 30), (50, 50, 50), -1)
        cv2.rectangle(ann, (10, 10), (10 + bar_w, 30), color, -1)

        # Text overlays
        cv2.putText(ann, f"Level : {result['fatigue_level']}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(ann, f"Score : {result['fatigue_score']:.1f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
        cv2.putText(ann, f"EAR   : {result['ear']:.2f}",
                    (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(ann, f"MAR   : {result['mar']:.2f}",
                    (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.putText(ann, f"Pitch : {result['pitch']:.1f}°",
                    (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if result["fatigue_type"]:
            cv2.putText(ann, " | ".join(result["fatigue_type"]),
                        (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return ann

    def get_session_summary(self):
        return {
            "total_frames": self.frame_count,
            "fatigue_log": self.fatigue_log,
        }

    def release(self):
        self.face_mesh.close()

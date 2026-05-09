# ============================================================
#  inference.py  —  PyTorch Fatigue Detection Engine
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import math

from config import *
from models import build_model, get_device


# Support both old (<=0.10.9) and new MediaPipe API
try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    mp_face_mesh = None
    print("Warning: mediapipe.solutions not available. Install mediapipe==0.10.9")


# ──────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ──────────────────────────────────────────────────────────────
def _euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def compute_ear(landmarks, eye_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    vert1 = _euclidean(pts[1], pts[5])
    vert2 = _euclidean(pts[2], pts[4])
    horiz = _euclidean(pts[0], pts[3])
    return (vert1 + vert2) / (2.0 * horiz + 1e-6)

def compute_mar(landmarks, mouth_indices, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in mouth_indices]
    return _euclidean(pts[2], pts[6]) / (_euclidean(pts[0], pts[4]) + 1e-6)

def compute_pitch(landmarks, w, h):
    model_pts = np.array([
        [0.0,0.0,0.0],[0.0,-63.6,-12.5],[-43.3,32.7,-26.0],
        [43.3,32.7,-26.0],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]
    ], dtype=np.float64)
    img_pts = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in [1, 152, 263, 33, 287, 57]
    ], dtype=np.float64)
    fl = w
    cam = np.array([[fl,0,w/2],[0,fl,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(model_pts, img_pts, cam, np.zeros((4,1)),
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    angles, *_ = cv2.RQDecomp3x3(rmat)
    return angles[0]


# ──────────────────────────────────────────────────────────────
# ROI EXTRACTION
# ──────────────────────────────────────────────────────────────
def extract_roi(frame, landmarks, indices, w, h, padding=20):
    pts = np.array([(int(landmarks[i].x*w), int(landmarks[i].y*h)) for i in indices])
    x1 = max(0,  pts[:,0].min() - padding)
    x2 = min(w,  pts[:,0].max() + padding)
    y1 = max(0,  pts[:,1].min() - padding)
    y2 = min(h,  pts[:,1].max() + padding)
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None

def extract_face_roi(frame, landmarks, w, h, padding=30):
    all_pts = np.array([(int(lm.x*w), int(lm.y*h)) for lm in landmarks])
    x1 = max(0, all_pts[:,0].min() - padding)
    x2 = min(w, all_pts[:,0].max() + padding)
    y1 = max(0, all_pts[:,1].min() - padding)
    y2 = min(h, all_pts[:,1].max() + padding)
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


# ──────────────────────────────────────────────────────────────
# PREPROCESS FOR PYTORCH MODEL
# ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_roi(roi):
    """BGR numpy → normalized tensor (1, 3, H, W)"""
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    return preprocess(rgb).unsqueeze(0)


# ──────────────────────────────────────────────────────────────
# FATIGUE ENGINE
# ──────────────────────────────────────────────────────────────
class FatigueEngine:
    def __init__(self, model_path: str):
        self.device = get_device()

        # Load PyTorch model
        self.model = build_model(freeze_backbone=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ Model loaded from {model_path}")

        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=FACE_CONFIDENCE,
            min_tracking_confidence=LANDMARK_CONFIDENCE
        )

        # Temporal smoothing
        self.score_history = deque(maxlen=TEMPORAL_WINDOW)
        self.ear_history   = deque(maxlen=TEMPORAL_WINDOW)
        self.frame_count   = 0

    def _predict(self, roi):
        """Run model on a single ROI → drowsy probability (0-1)"""
        tensor = preprocess_roi(roi).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob   = F.softmax(logits, dim=1)[0][1].item()  # prob of class 1 = drowsy
        return prob

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1
        h, w = frame.shape[:2]

        result = {
            "frame":          frame.copy(),
            "face_detected":  False,
            "ear":            0.0,
            "mar":            0.0,
            "pitch":          0.0,
            "drowsy_prob":    0.0,
            "fatigue_score":  0.0,
            "fatigue_level":  "ALERT",
        }

        rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = self.face_mesh.process(rgb)

        if not mesh_result.multi_face_landmarks:
            return result

        result["face_detected"] = True
        lm = mesh_result.multi_face_landmarks[0].landmark

        # ── Geometric features ────────────────────────────────
        ear   = (compute_ear(lm, LEFT_EYE_IDX,  w, h) +
                 compute_ear(lm, RIGHT_EYE_IDX, w, h)) / 2.0
        mar   = compute_mar(lm, MOUTH_IDX, w, h)
        pitch = compute_pitch(lm, w, h)

        result["ear"]   = ear
        result["mar"]   = mar
        result["pitch"] = pitch
        self.ear_history.append(ear)

        # ── CNN prediction on face ROI ────────────────────────
        face_roi = extract_face_roi(frame, lm, w, h)
        if face_roi is not None:
            drowsy_prob = self._predict(face_roi)
        else:
            # fallback to geometric only
            drowsy_prob = float(ear < EAR_THRESHOLD or mar > MAR_THRESHOLD)

        result["drowsy_prob"] = drowsy_prob

        # ── PERCLOS ───────────────────────────────────────────
        perclos = (np.array(self.ear_history) < EAR_THRESHOLD).mean() \
                  if self.ear_history else 0.0

        # ── Score fusion ──────────────────────────────────────
        # CNN carries most weight, geometric features support it
        score = (
            drowsy_prob             * 60 +   # CNN output    60%
            perclos                 * 25 +   # eye closure   25%
            float(mar > MAR_THRESHOLD) * 10 + # yawning      10%
            float(abs(pitch) > PITCH_THRESHOLD) * 5  # head  5%
        )
        self.score_history.append(score)

        # Temporal smoothing
        smooth_score = np.mean(self.score_history)
        result["fatigue_score"] = smooth_score

        # ── Level ─────────────────────────────────────────────
        if smooth_score < 30:
            result["fatigue_level"] = "ALERT"
        elif smooth_score < 60:
            result["fatigue_level"] = "WARNING"
        else:
            result["fatigue_level"] = "DANGER"

        # ── Annotate frame ────────────────────────────────────
        result["frame"] = self._annotate(frame, result)
        return result

    def _annotate(self, frame, result):
        ann   = frame.copy()
        score = result["fatigue_score"]
        level = result["fatigue_level"]
        color = {"ALERT": (0,200,0), "WARNING": (0,200,255), "DANGER": (0,0,255)}[level]

        bar_w = int(score / 100 * 250)
        cv2.rectangle(ann, (10,10), (260,30), (50,50,50), -1)
        cv2.rectangle(ann, (10,10), (10+bar_w,30), color, -1)

        cv2.putText(ann, f"{level}  {score:.0f}%",
                    (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(ann, f"EAR:{result['ear']:.2f}  MAR:{result['mar']:.2f}  Pitch:{result['pitch']:.1f}",
                    (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        return ann

    def release(self):
        self.face_mesh.close()

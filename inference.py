# ============================================================
#  inference.py  —  PyTorch Fatigue Detection Engine
#  Compatible with mediapipe >= 0.10.30 (new Tasks API)
# ============================================================

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import math
import urllib.request
import os

from config import *
from models import build_model, get_device


# ──────────────────────────────────────────────────────────────
# MEDIAPIPE  —  new Tasks API (0.10.30+)
# ──────────────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

# Download the face landmarker model if not present
LANDMARKER_PATH = "face_landmarker.task"

def _download_landmarker():
    if not os.path.exists(LANDMARKER_PATH):
        print("Downloading MediaPipe face landmarker model...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, LANDMARKER_PATH)
        print("  ✓ Downloaded face_landmarker.task")

_download_landmarker()


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
    fl  = float(w)
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
def extract_face_roi(frame, landmarks, w, h, padding=30):
    all_pts = np.array([(int(lm.x*w), int(lm.y*h)) for lm in landmarks])
    x1 = max(0, all_pts[:,0].min() - padding)
    x2 = min(w, all_pts[:,0].max() + padding)
    y1 = max(0, all_pts[:,1].min() - padding)
    y2 = min(h, all_pts[:,1].max() + padding)
    roi = frame[y1:y2, x1:x2]
    return roi if roi.size > 0 else None


# ──────────────────────────────────────────────────────────────
# PREPROCESS FOR PYTORCH
# ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def preprocess_roi(roi):
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
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"  ✓ Model loaded: {model_path}")

        # MediaPipe Face Landmarker (new Tasks API)
        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=FACE_CONFIDENCE,
            min_face_presence_confidence=FACE_CONFIDENCE,
            min_tracking_confidence=LANDMARK_CONFIDENCE,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # Buffers
        self.score_history = deque(maxlen=TEMPORAL_WINDOW)
        self.ear_history   = deque(maxlen=TEMPORAL_WINDOW)
        self.frame_count   = 0

    def _get_landmarks(self, frame):
        """Run face landmarker → returns landmark list or None"""
        rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result    = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]   # list of NormalizedLandmark

    def _predict(self, roi):
        tensor = preprocess_roi(roi).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob   = F.softmax(logits, dim=1)[0][1].item()
        return prob

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1
        h, w = frame.shape[:2]

        result = {
            "frame":         frame.copy(),
            "face_detected": False,
            "ear":           0.0,
            "mar":           0.0,
            "pitch":         0.0,
            "drowsy_prob":   0.0,
            "fatigue_score": 0.0,
            "fatigue_level": "ALERT",
        }

        landmarks = self._get_landmarks(frame)
        if landmarks is None:
            return result

        result["face_detected"] = True

        ear   = (compute_ear(landmarks, LEFT_EYE_IDX,  w, h) +
                 compute_ear(landmarks, RIGHT_EYE_IDX, w, h)) / 2.0
        mar   = compute_mar(landmarks, MOUTH_IDX, w, h)
        pitch = compute_pitch(landmarks, w, h)

        result.update({"ear": ear, "mar": mar, "pitch": pitch})
        self.ear_history.append(ear)

        # CNN prediction
        face_roi    = extract_face_roi(frame, landmarks, w, h)
        drowsy_prob = self._predict(face_roi) if face_roi is not None \
                      else float(ear < EAR_THRESHOLD or mar > MAR_THRESHOLD)
        result["drowsy_prob"] = drowsy_prob

        # PERCLOS
        perclos = (np.array(self.ear_history) < EAR_THRESHOLD).mean() \
                  if self.ear_history else 0.0

        # Score fusion
        score = (
            drowsy_prob                          * 60 +
            perclos                              * 25 +
            float(mar > MAR_THRESHOLD)           * 10 +
            float(abs(pitch) > PITCH_THRESHOLD)  *  5
        )
        self.score_history.append(score)
        smooth = np.mean(self.score_history)
        result["fatigue_score"] = smooth
        result["fatigue_level"] = (
            "DANGER"  if smooth >= 60 else
            "WARNING" if smooth >= 30 else
            "ALERT"
        )

        result["landmarks"] = landmarks
        result["frame"] = self._annotate(frame, result)
        return result

    def _annotate(self, frame, result):
        ann   = frame.copy()
        h, w  = ann.shape[:2]
        score = result["fatigue_score"]
        level = result["fatigue_level"]
        color = {"ALERT":(0,200,0),"WARNING":(0,200,255),"DANGER":(0,0,255)}[level]
        landmarks = result.get("landmarks", None)

        # ── Draw all 468 face landmarks ───────────────────────
        if landmarks:
            # Full mesh — tiny dots for all points
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(ann, (x, y), 1, (0, 180, 180), -1)

            # Eye landmarks — highlighted in accent color
            for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(ann, (x, y), 3, (0, 229, 255), -1)

            # Connect eye points with lines
            for eye_idx in [LEFT_EYE_IDX, RIGHT_EYE_IDX]:
                pts = np.array([
                    (int(landmarks[i].x * w), int(landmarks[i].y * h))
                    for i in eye_idx
                ], dtype=np.int32)
                cv2.polylines(ann, [pts], isClosed=True, color=(0, 229, 255), thickness=1)

            # Mouth landmarks — highlighted in yellow
            for idx in MOUTH_IDX:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(ann, (x, y), 3, (0, 215, 255), -1)

            # Connect mouth points
            mouth_pts = np.array([
                (int(landmarks[i].x * w), int(landmarks[i].y * h))
                for i in MOUTH_IDX
            ], dtype=np.int32)
            cv2.polylines(ann, [mouth_pts], isClosed=True, color=(0, 215, 255), thickness=1)

        # ── HUD overlay ───────────────────────────────────────
        # Score bar background
        cv2.rectangle(ann, (10, 10), (260, 30), (30, 30, 30), -1)
        bar_w = int(score / 100 * 250)
        cv2.rectangle(ann, (10, 10), (10 + bar_w, 30), color, -1)
        cv2.rectangle(ann, (10, 10), (260, 30), (80, 80, 80), 1)

        # Level + score text
        cv2.putText(ann, f"{level}  {score:.0f}%",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Metrics
        cv2.putText(ann,
                    f"EAR:{result['ear']:.2f}  MAR:{result['mar']:.2f}  Pitch:{result['pitch']:.1f}deg",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        # Drowsy probability bar (small, bottom left)
        prob = result.get("drowsy_prob", 0)
        prob_w = int(prob * 120)
        cv2.putText(ann, "DROWSY PROB", (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
        cv2.rectangle(ann, (10, h - 25), (130, h - 12), (30, 30, 30), -1)
        cv2.rectangle(ann, (10, h - 25), (10 + prob_w, h - 12), color, -1)

        return ann

    def release(self):
        self.landmarker.close()

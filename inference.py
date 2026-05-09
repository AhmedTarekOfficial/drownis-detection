# -*- coding: utf-8 -*-
# ============================================================
# inference.py — PyTorch Fatigue Detection Engine

import os
import math
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

from config import *
from models import build_model, get_device


# ============================================================
# DOWNLOAD LANDMARK MODEL
# ============================================================

LANDMARKER_PATH = "face_landmarker.task"


def download_landmarker():
    if os.path.exists(LANDMARKER_PATH):
        return

    print("Downloading MediaPipe face landmarker...")

    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    )

    urllib.request.urlretrieve(url, LANDMARKER_PATH)

    print("✓ Download complete")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


# ============================================================
# GEOMETRY HELPERS
# ============================================================


def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)



def landmarks_to_points(landmarks, indices, w, h):
    return [
        (
            int(landmarks[i].x * w),
            int(landmarks[i].y * h),
        )
        for i in indices
    ]



def compute_ear(landmarks, eye_indices, w, h):
    pts = landmarks_to_points(landmarks, eye_indices, w, h)

    vertical_1 = euclidean(pts[1], pts[5])
    vertical_2 = euclidean(pts[2], pts[4])
    horizontal = euclidean(pts[0], pts[3])

    return (vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6)



def compute_mar(landmarks, mouth_indices, w, h):
    pts = landmarks_to_points(landmarks, mouth_indices, w, h)

    vertical = euclidean(pts[2], pts[6])
    horizontal = euclidean(pts[0], pts[4])

    return vertical / (horizontal + 1e-6)



def compute_pitch(landmarks, w, h):

    model_points = np.array([
        [0.0, 0.0, 0.0],
        [0.0, -63.6, -12.5],
        [-43.3, 32.7, -26.0],
        [43.3, 32.7, -26.0],
        [-28.9, -28.9, -24.1],
        [28.9, -28.9, -24.1],
    ], dtype=np.float64)

    image_points = np.array([
        (landmarks[i].x * w, landmarks[i].y * h)
        for i in [1, 152, 263, 33, 287, 57]
    ], dtype=np.float64)

    focal_length = float(w)

    camera_matrix = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1],
    ], dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return 0.0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)

    return angles[0]


# ============================================================
# DRAWING
# ============================================================


def draw_landmarks(frame, landmarks, w, h):

    # Full mesh
    for lm in landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)

        cv2.circle(frame, (x, y), 1, (0, 180, 180), -1)

    # Eyes
    for idx in LEFT_EYE_IDX + RIGHT_EYE_IDX:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        cv2.circle(frame, (x, y), 3, (0, 229, 255), -1)

    left_eye = np.array(
        landmarks_to_points(landmarks, LEFT_EYE_IDX, w, h),
        dtype=np.int32,
    )

    right_eye = np.array(
        landmarks_to_points(landmarks, RIGHT_EYE_IDX, w, h),
        dtype=np.int32,
    )

    cv2.polylines(frame, [left_eye], True, (0, 229, 255), 1)
    cv2.polylines(frame, [right_eye], True, (0, 229, 255), 1)

    # Mouth
    for idx in MOUTH_IDX:
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)

        cv2.circle(frame, (x, y), 3, (0, 215, 255), -1)

    mouth = np.array(
        landmarks_to_points(landmarks, MOUTH_IDX, w, h),
        dtype=np.int32,
    )

    cv2.polylines(frame, [mouth], True, (0, 215, 255), 1)



def draw_hud(frame, result):

    h, w = frame.shape[:2]

    score = result["fatigue_score"]
    level = result["fatigue_level"]

    colors = {
        "ALERT": (0, 200, 0),
        "WARNING": (0, 200, 255),
        "DANGER": (0, 0, 255),
    }

    color = colors[level]

    # Main score bar
    cv2.rectangle(frame, (10, 10), (260, 30), (30, 30, 30), -1)

    bar_width = int(score / 100 * 250)

    cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), color, -1)

    cv2.rectangle(frame, (10, 10), (260, 30), (80, 80, 80), 1)

    cv2.putText(
        frame,
        f"{level}  {score:.0f}%",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    cv2.putText(
        frame,
        (
            f"EAR:{result['ear']:.2f}  "
            f"MAR:{result['mar']:.2f}  "
            f"Pitch:{result['pitch']:.1f}"
        ),
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Drowsiness probability
    prob = result["drowsy_prob"]

    cv2.putText(
        frame,
        "DROWSY PROB",
        (10, h - 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (120, 120, 120),
        1,
    )

    cv2.rectangle(frame, (10, h - 25), (130, h - 12), (30, 30, 30), -1)

    cv2.rectangle(
        frame,
        (10, h - 25),
        (10 + int(prob * 120), h - 12),
        color,
        -1,
    )


# ============================================================
# ROI EXTRACTION
# ============================================================


def extract_face_roi(frame, landmarks, w, h, padding=30):

    points = np.array([
        (int(lm.x * w), int(lm.y * h))
        for lm in landmarks
    ])

    x1 = max(0, points[:, 0].min() - padding)
    x2 = min(w, points[:, 0].max() + padding)

    y1 = max(0, points[:, 1].min() - padding)
    y2 = min(h, points[:, 1].max() + padding)

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None

    return roi



def preprocess_roi(roi):

    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    tensor = preprocess(rgb)

    return tensor.unsqueeze(0)


# ============================================================
# FATIGUE ENGINE
# ============================================================


class FatigueEngine:

    def __init__(self, model_path):

        download_landmarker()

        self.device = get_device()

        self.model = self.load_model(model_path)

        self.landmarker = self.create_landmarker()

        self.score_history = deque(maxlen=TEMPORAL_WINDOW)
        self.ear_history = deque(maxlen=TEMPORAL_WINDOW)

    # ========================================================
    # MODEL
    # ========================================================

    def load_model(self, model_path):

        model = build_model(freeze_backbone=False)

        model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        model.to(self.device)
        model.eval()

        print(f"✓ Model loaded: {model_path}")

        return model

    # ========================================================
    # MEDIAPIPE
    # ========================================================

    def create_landmarker(self):

        options = FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=LANDMARKER_PATH
            ),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=FACE_CONFIDENCE,
            min_face_presence_confidence=FACE_CONFIDENCE,
            min_tracking_confidence=LANDMARK_CONFIDENCE,
        )

        return FaceLandmarker.create_from_options(options)

    # ========================================================
    # LANDMARK DETECTION
    # ========================================================

    def get_landmarks(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        return result.face_landmarks[0]

    # ========================================================
    # CNN PREDICTION
    # ========================================================

    def predict_batch(self, rois):

        tensors = [preprocess_roi(roi) for roi in rois]

        batch = torch.cat(tensors, dim=0).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)[:, 1]

        return probs.cpu().tolist()

    # ========================================================
    # SCORE CALCULATION
    # ========================================================

    def compute_score(self, prob, ear, mar, pitch):

        perclos = (
            (np.array(self.ear_history) < EAR_THRESHOLD).mean()
            if self.ear_history
            else 0.0
        )

        score = (
            prob * 60
            + perclos * 25
            + float(mar > MAR_THRESHOLD) * 10
            + float(abs(pitch) > PITCH_THRESHOLD) * 5
        )

        self.score_history.append(score)

        return float(np.mean(self.score_history))

    # ========================================================
    # LEVEL CLASSIFICATION
    # ========================================================

    @staticmethod
    def classify_level(score):

        if score >= 60:
            return "DANGER"

        if score >= 30:
            return "WARNING"

        return "ALERT"

    # ========================================================
    # PROCESS SINGLE FRAME
    # ========================================================

    def process_frame(self, frame):

        results = self.process_batch([frame])

        return results[0]

    # ========================================================
    # PROCESS BATCH
    # ========================================================

    def process_batch(self, frames):

        if not frames:
            return []

        h, w = frames[0].shape[:2]

        rois = []
        temp_results = []

        # ----------------------------------------------------
        # STEP 1 — LANDMARKS + FEATURES
        # ----------------------------------------------------

        for frame in frames:

            annotated = frame.copy()

            result = {
                "frame": annotated,
                "face_detected": False,
                "ear": 0.0,
                "mar": 0.0,
                "pitch": 0.0,
                "drowsy_prob": 0.0,
                "fatigue_score": 0.0,
                "fatigue_level": "ALERT",
            }

            landmarks = self.get_landmarks(frame)

            if landmarks is None:
                draw_hud(annotated, result)
                temp_results.append(result)

                rois.append(np.zeros((*IMG_SIZE, 3), np.uint8))
                continue

            result["face_detected"] = True

            draw_landmarks(annotated, landmarks, w, h)

            ear = (
                compute_ear(landmarks, LEFT_EYE_IDX, w, h)
                + compute_ear(landmarks, RIGHT_EYE_IDX, w, h)
            ) / 2.0

            mar = compute_mar(landmarks, MOUTH_IDX, w, h)

            pitch = compute_pitch(landmarks, w, h)

            self.ear_history.append(ear)

            result["ear"] = ear
            result["mar"] = mar
            result["pitch"] = pitch

            roi = extract_face_roi(frame, landmarks, w, h)

            if roi is None:
                roi = np.zeros((*IMG_SIZE, 3), np.uint8)

            rois.append(roi)
            temp_results.append(result)

        # ----------------------------------------------------
        # STEP 2 — CNN BATCH PREDICTION
        # ----------------------------------------------------

        probs = self.predict_batch(rois)

        # ----------------------------------------------------
        # STEP 3 — FINAL RESULTS
        # ----------------------------------------------------

        final_results = []

        for result, prob in zip(temp_results, probs):

            if not result["face_detected"]:
                final_results.append(result)
                continue

            result["drowsy_prob"] = prob

            score = self.compute_score(
                prob,
                result["ear"],
                result["mar"],
                result["pitch"],
            )

            result["fatigue_score"] = score
            result["fatigue_level"] = self.classify_level(score)

            draw_hud(result["frame"], result)

            final_results.append(result)

        return final_results

    # ========================================================
    # CLEANUP
    # ========================================================

    def release(self):
        self.landmarker.close()



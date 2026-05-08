# ============================================================
#  config.py  —  Central configuration (drowsy / undrowsy)
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "/content/drownis-detection/data/driver_drowsiness_dataset/Driver")     # ← ضع مسار داتاك هنا
MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
LOGS_DIR   = os.path.join(BASE_DIR, "logs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR,   exist_ok=True)

# ── Dataset — just two folders ───────────────────────────────
#   data/
#   ├── drowsy/
#   └── undrowsy/
CLASS_NAMES = ["Non_Drowsy", "Drowsy"]   # 0 = undrowsy, 1 = drowsy

# ── Image settings ───────────────────────────────────────────
IMG_SIZE     = (224, 224)
IMG_CHANNELS = 3
INPUT_SHAPE  = (*IMG_SIZE, IMG_CHANNELS)

# ── Training hyperparameters ─────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5
VAL_SPLIT     = 0.15
TEST_SPLIT    = 0.15
SEED          = 42

# ── Augmentation ─────────────────────────────────────────────
AUGMENT = dict(
    rotation_range     = 15,
    width_shift_range  = 0.1,
    height_shift_range = 0.1,
    zoom_range         = 0.15,
    horizontal_flip    = True,
    brightness_range   = [0.7, 1.3],
    fill_mode          = "nearest",
)

# ── MediaPipe ────────────────────────────────────────────────
FACE_CONFIDENCE     = 0.5
LANDMARK_CONFIDENCE = 0.5

LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX     = [61, 291, 81, 178, 13, 14, 311, 402]

# ── Thresholds (Inference) ────────────────────────────────────
EAR_THRESHOLD     = 0.25
MAR_THRESHOLD     = 0.6
PITCH_THRESHOLD   = 20
PERCLOS_THRESHOLD = 0.7
TEMPORAL_WINDOW   = 60
MICROSLEEP_FRAMES = 30

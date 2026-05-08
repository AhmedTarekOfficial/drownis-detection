# ============================================================
#  data_loader.py  —  loads from drowsy / undrowsy folders
# ============================================================

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *


def load_dataset(img_size=IMG_SIZE):
    """
    Reads images from:
        data/drowsy/    → label 1
        data/undrowsy/  → label 0
    """
    X, y = [], []

    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(
                f"Folder not found: {class_dir}\n"
                f"Make sure your data folder contains: drowsy/ and undrowsy/"
            )

        images = [f for f in os.listdir(class_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        print(f"  [{class_name}]  {len(images)} images")

        for img_file in images:
            img = cv2.imread(os.path.join(class_dir, img_file))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int32)
    print(f"\n  Total: {len(X)} images — classes: {CLASS_NAMES}")
    return X, y


def split_dataset(X, y):
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y)

    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_ratio, random_state=SEED, stratify=y_tv)

    print(f"  Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}\n")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_generators(X_train, y_train, X_val, y_val):
    train_gen = ImageDataGenerator(**AUGMENT)
    val_gen   = ImageDataGenerator()

    y_train_cat = tf.keras.utils.to_categorical(y_train, 2)
    y_val_cat   = tf.keras.utils.to_categorical(y_val,   2)

    return (
        train_gen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE, shuffle=True, seed=SEED),
        val_gen.flow(X_val,   y_val_cat,   batch_size=BATCH_SIZE, shuffle=False),
    )


def compute_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

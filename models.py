# ============================================================
#  models.py  —  PyTorch CNN for drowsy / undrowsy
# ============================================================

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, freeze_backbone=True):
    """
    MobileNetV2 pretrained on ImageNet
    → replace the final classifier for binary fatigue detection
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )

    return model


def unfreeze_top_layers(model, n_layers=20):
    """Unfreeze last N layers of the backbone for fine-tuning"""
    all_params = list(model.features.parameters())
    for param in all_params[-n_layers:]:
        param.requires_grad = True
    unfrozen = sum(p.requires_grad for p in model.parameters())
    print(f"  Trainable params after unfreeze: {unfrozen:,}")
    return model


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    return device

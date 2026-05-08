# ============================================================
#  train.py  —  PyTorch Training Pipeline
#  Usage:  python train.py
#          or in Colab: run all cells
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

from config import *
from models import build_model, unfreeze_top_layers, get_device


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class FatigueDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Reads from:
            root_dir/drowsy/
            root_dir/undrowsy/
        """
        self.samples   = []
        self.transform = transform

        for label, class_name in enumerate(CLASS_NAMES):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(
                    f"Folder not found: {class_dir}\n"
                    f"Expected: drowsy/ and undrowsy/ inside {root_dir}"
                )
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(class_dir, fname), label))

        print(f"  Total samples loaded: {len(self.samples)}")
        drowsy_n   = sum(1 for _, l in self.samples if l == 1)
        undrowsy_n = sum(1 for _, l in self.samples if l == 0)
        print(f"  drowsy: {drowsy_n}  |  undrowsy: {undrowsy_n}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = outputs.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correct / total


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs     = model(imgs)
            loss        = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────
def run_training(model, train_loader, val_loader, device,
                 epochs=EPOCHS, lr=LEARNING_RATE, tag="phase1"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    save_path = os.path.join(MODELS_DIR, f"fatigue_{tag}.pth")

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {tr_loss:.4f}  Acc: {tr_acc:.3f} | "
              f"Val Loss: {vl_loss:.4f}  Acc: {vl_acc:.3f}")

        # Save best
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 7:
                print("  Early stopping triggered.")
                break

    model.load_state_dict(torch.load(save_path))
    print(f"  ✓ Best model saved → {save_path}")
    return model, history


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────
def final_evaluation(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n  Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    path = os.path.join(LOGS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"  ✓ Confusion matrix saved → {path}")


# ──────────────────────────────────────────────────────────────
# PLOT HISTORY
# ──────────────────────────────────────────────────────────────
def plot_history(h1, h2=None):
    loss = h1["train_loss"] + (h2["train_loss"] if h2 else [])
    val  = h1["val_loss"]   + (h2["val_loss"]   if h2 else [])
    acc  = h1["train_acc"]  + (h2["train_acc"]  if h2 else [])
    vacc = h1["val_acc"]    + (h2["val_acc"]    if h2 else [])
    ep   = range(1, len(loss) + 1)
    split = len(h1["train_loss"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Fatigue Model — Training History")

    axes[0].plot(ep, loss, label="Train"); axes[0].plot(ep, val, label="Val")
    if h2: axes[0].axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(ep, acc, label="Train"); axes[1].plot(ep, vacc, label="Val")
    if h2: axes[1].axvline(split, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Accuracy"); axes[1].legend()

    plt.tight_layout()
    path = os.path.join(LOGS_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    print(f"  ✓ Training plot saved → {path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()

    # ── Load full dataset ────────────────────────────────────
    print("\n=== Loading Dataset ===")
    full_dataset = FatigueDataset(DATA_DIR, transform=train_transform)
    n = len(full_dataset)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_test - n_val

    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    # Val and test use no augmentation
    val_ds.dataset.transform  = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"  Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

    # ── Phase 1: Frozen backbone ─────────────────────────────
    print("\n=== Phase 1 — Frozen Backbone ===")
    model = build_model(freeze_backbone=True).to(device)
    model, h1 = run_training(model, train_loader, val_loader, device,
                              epochs=EPOCHS, lr=LEARNING_RATE, tag="phase1")

    # ── Phase 2: Fine-tune ───────────────────────────────────
    print("\n=== Phase 2 — Fine-tuning ===")
    model = unfreeze_top_layers(model, n_layers=20)
    model, h2 = run_training(model, train_loader, val_loader, device,
                              epochs=15, lr=FINE_TUNE_LR, tag="phase2")

    # ── Evaluate ─────────────────────────────────────────────
    print("\n=== Final Evaluation on Test Set ===")
    final_evaluation(model, test_loader, device)
    plot_history(h1, h2)

    print("\n✓ Training complete!")

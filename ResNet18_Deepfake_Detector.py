import os
import glob
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from torchvision.models import ResNet18_Weights

from sklearn.metrics import classification_report, confusion_matrix


# ─────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    def __init__(self, data_root, split="train", frames_per_sample=8):
        self.data_root = Path(data_root)
        self.split = split
        self.frames_per_sample = frames_per_sample

        # Load images
        real = list((self.data_root / "real_data").glob("*.jpg"))
        fake = list((self.data_root / "fake_data").glob("*.jpg"))

        self.samples = [(p, 0) for p in real] + [(p, 1) for p in fake]

        print(f"[INFO] {split} split loaded: {len(self.samples)} samples")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(label).long()


# ─────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.backbone(x)


# ─────────────────────────────────────────────
# TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0
    running_corrects = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(1)
        running_corrects += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_corrects / total


# ─────────────────────────────────────────────
# VALIDATION FUNCTION
# ─────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    running_loss = 0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            running_corrects += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, running_corrects / total


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="Dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    train_ds = DeepfakeDataset(args.data_root, "train")
    val_ds = DeepfakeDataset(args.data_root, "val")
    test_ds = DeepfakeDataset(args.data_root, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = DeepfakeDetector().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"[EPOCH {epoch+1}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ─────────────────────────────────────────────
    # FINAL TEST EVALUATION
    # ─────────────────────────────────────────────
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\n[TEST] Loss: {test_loss:.4f} | Acc: {test_acc * 100:.2f}%")

    # ─────────────────────────────────────────────
    # DETAILED METRICS: Classification Report & Confusion Matrix
    # ─────────────────────────────────────────────
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

    print("===== CONFUSION MATRIX =====")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)


if __name__ == "__main__":
    main()

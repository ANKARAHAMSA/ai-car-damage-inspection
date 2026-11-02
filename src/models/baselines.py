import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils import load_config, seed_everything
from src.data.dataset import CarDamageClassificationDataset

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SmallCNN(nn.Module):
    """Baseline 2: Small 3-layer CNN trained from scratch (no transfer learning)."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.classifier(x).squeeze(-1)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "AUROC": round(roc_auc_score(y_true, y_prob), 4),
    }

def train_and_eval(model, train_loader, val_loader, epochs, lr, device, pos_weight):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_auroc = 0.0
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(y.numpy())

        m = compute_metrics(np.array(val_targets), np.array(val_probs))
        if m["AUROC"] >= best_auroc:
            best_auroc = m["AUROC"]
            best_metrics = m
        print(f"   Epoch {epoch:2d}/{epochs} | Val AUROC: {m['AUROC']:.4f} | F1: {m['F1']:.4f} | Recall: {m['Recall']:.4f}")

    return best_metrics

def run_baselines(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    device = get_device()
    print(f"🚀 Running Phase 3 Baselines on device: {device}")

    raw_dir = cfg["data"]["raw_dir"]
    train_ds = CarDamageClassificationDataset("data/splits/train_meta.csv", raw_dir, img_size=224, is_train=False)
    val_ds = CarDamageClassificationDataset("data/splits/val_meta.csv", raw_dir, img_size=224, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Calculate positive class weight (undamaged / damaged ~ 2.33)
    num_pos = train_ds.df["is_damaged"].sum()
    num_neg = len(train_ds.df) - num_pos
    pos_weight = float(num_neg / max(1, num_pos))

    results = []

    # 1. Majority Class Baseline (predicts 0 = undamaged for everything)
    print("\n1️⃣ Evaluating Baseline 1: Majority Class...")
    y_val = val_ds.df["is_damaged"].values
    maj_probs = np.zeros_like(y_val, dtype=float)
    m1 = compute_metrics(y_val, maj_probs)
    m1["Baseline"] = "Majority Class (Always Undamaged)"
    results.append(m1)
    print(f"   Result: {m1}")

    # 2. Small CNN from scratch
    print("\n2️⃣ Training Baseline 2: Small CNN from Scratch (8 epochs)...")
    seed_everything(cfg["seed"])
    cnn = SmallCNN()
    m2 = train_and_eval(cnn, train_loader, val_loader, epochs=8, lr=1e-3, device=device, pos_weight=pos_weight)
    m2["Baseline"] = "Small CNN from Scratch"
    results.append(m2)

    # 3. EfficientNet-B0 (Frozen Backbone)
    print("\n3️⃣ Training Baseline 3: EfficientNet-B0 (Frozen Backbone, 5 epochs)...")
    seed_everything(cfg["seed"])
    effnet = timm.create_model("efficientnet_b0", pretrained=True, num_classes=1)
    for name, param in effnet.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    class EffWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x).squeeze(-1)

    m3 = train_and_eval(EffWrapper(effnet), train_loader, val_loader, epochs=5, lr=1e-3, device=device, pos_weight=pos_weight)
    m3["Baseline"] = "EfficientNet-B0 (Frozen Backbone)"
    results.append(m3)

    df_res = pd.DataFrame(results)[["Baseline", "Accuracy", "Precision", "Recall", "F1", "AUROC"]]
    os.makedirs("reports", exist_ok=True)
    df_res.to_csv("reports/phase3_baselines.csv", index=False)

    md_lines = [
        "# Phase 3: Baseline Models Comparison (Validation Set)\n",
        "| Baseline | Accuracy | Precision | Recall | F1 | AUROC |",
        "|---|---|---|---|---|---|"
    ]
    for _, r in df_res.iterrows():
        md_lines.append(f"| {r['Baseline']} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | {r['Recall']:.4f} | {r['F1']:.4f} | {r['AUROC']:.4f} |")

    with open("reports/phase3_baselines.md", "w") as f:
        f.write("\n".join(md_lines))

    print("\n✅ Phase 3 Baselines Complete! Summary Table:")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    run_baselines()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from src.utils import load_config, seed_everything
from src.data.dataset import CarDamageClassificationDataset
from src.models.baselines import get_device

class DamageClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
        self.calib_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.calib_bias = nn.Parameter(torch.zeros(1), requires_grad=False)

    def freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name and "head" not in name:
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x).squeeze(-1)

    def calibrated_prob(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits * self.calib_scale + self.calib_bias)

def compute_ece(probs, targets, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_accs, bin_confs = [], []
    for i in range(n_bins):
        in_bin = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(targets[in_bin])
            conf_in_bin = np.mean(probs[in_bin])
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
    return float(ece), bin_confs, bin_accs

def fit_calibration(logits_np, targets_np):
    """Fits temperature scale + bias correction (Platt scaling) on validation logits to remove pos_weight shift."""
    lr = LogisticRegression(C=1e5, solver="lbfgs")
    lr.fit(logits_np.reshape(-1, 1), targets_np)
    scale = float(lr.coef_[0][0])
    bias = float(lr.intercept_[0])
    probs_cal = lr.predict_proba(logits_np.reshape(-1, 1))[:, 1]
    return scale, bias, probs_cal

def plot_reliability_diagram(probs_before, probs_after, targets, ece_before, ece_after, out_path):
    _, conf_b, acc_b = compute_ece(probs_before, targets)
    _, conf_a, acc_a = compute_ece(probs_after, targets)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, acc, title in [
        (axes[0], acc_b, f"Before Calibration (ECE = {ece_before:.4f})"),
        (axes[1], acc_a, f"After Calibration (ECE = {ece_after:.4f})")
    ]:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.bar(np.linspace(0.05, 0.95, 10), acc, width=0.08, alpha=0.75, edgecolor="black", label="Empirical Positives")
        ax.set_xlabel("Predicted Probability (Damaged)")
        ax.set_ylabel("Empirical Fraction Positives")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def run_single_experiment(model_name, img_size, use_aug, epochs_frozen, epochs_unfrozen, cfg, device):
    seed_everything(cfg["seed"])
    raw_dir = cfg["data"]["raw_dir"]
    train_ds = CarDamageClassificationDataset("data/splits/train_meta.csv", raw_dir, img_size=img_size, is_train=use_aug)
    val_ds = CarDamageClassificationDataset("data/splits/val_meta.csv", raw_dir, img_size=img_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=cfg["classifier"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["classifier"]["batch_size"], shuffle=False)

    num_pos = train_ds.df["is_damaged"].sum()
    num_neg = len(train_ds.df) - num_pos
    pos_weight = torch.tensor([float(num_neg / max(1, num_pos))], device=device)

    model = DamageClassifier(model_name=model_name, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    smoothing = cfg["classifier"]["label_smoothing"]

    model.freeze_backbone()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["classifier"]["lr_head"], weight_decay=1e-2)

    best_auroc = 0.0
    best_logits, best_targets, best_state = None, None, None
    total_epochs = epochs_frozen + epochs_unfrozen

    for epoch in range(1, total_epochs + 1):
        if epoch == epochs_frozen + 1:
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["classifier"]["lr_unfrozen"], weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_unfrozen, eta_min=1e-6)

        model.train()
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            y_smooth = y * (1.0 - smoothing) + 0.5 * smoothing
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y_smooth)
            loss.backward()
            optimizer.step()

        if epoch > epochs_frozen:
            scheduler.step()

        model.eval()
        val_logits, val_targets = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device)
                val_logits.extend(model(x).cpu().numpy())
                val_targets.extend(y.numpy())

        val_logits = np.array(val_logits)
        val_targets = np.array(val_targets)
        probs = 1.0 / (1.0 + np.exp(-val_logits))
        auroc = roc_auc_score(val_targets, probs)
        if auroc > best_auroc:
            best_auroc = auroc
            best_logits = val_logits.copy()
            best_targets = val_targets.copy()
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    probs_05 = (1.0 / (1.0 + np.exp(-best_logits)) >= 0.5).astype(int)
    f1_05 = f1_score(best_targets, probs_05, zero_division=0)
    print(f"   Done {model_name} ({img_size}px, aug={use_aug}) -> Best Val AUROC: {best_auroc:.4f} | F1@0.5: {f1_05:.4f}")
    return best_auroc, f1_05, best_logits, best_targets, best_state

def train_stage1(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    device = get_device()
    print(f"🚀 Running Phase 4 Classifier & Ablation Study on {device}...\n")

    experiments = [
        ("efficientnet_b0", 224, False, "EfficientNet-B0 (224px, No Aug)"),
        ("efficientnet_b0", 224, True,  "EfficientNet-B0 (224px, With Aug)"),
        ("efficientnet_b0", 320, True,  "EfficientNet-B0 (320px, With Aug)"),
        ("convnext_tiny",   224, True,  "ConvNeXt-Tiny (224px, With Aug)"),
    ]

    ablation_rows = []
    overall_best = (-1.0, None, None, None, None, None, None)

    for m_name, sz, aug, desc in experiments:
        print(f"▶ Training {desc}...")
        auroc, f1_05, logits, targets, state = run_single_experiment(
            m_name, sz, aug, epochs_frozen=4, epochs_unfrozen=12, cfg=cfg, device=device
        )
        ablation_rows.append({"Experiment": desc, "Model": m_name, "ImgSize": sz, "Augmentation": aug, "Val_AUROC": round(auroc, 4), "Val_F1@0.5": round(f1_05, 4)})
        if auroc > overall_best[0]:
            overall_best = (auroc, m_name, sz, desc, logits, targets, state)

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv("reports/phase4_ablations.csv", index=False)
    print("\n📊 Phase 4 Ablation Results:")
    print(df_abl.to_string(index=False))

    best_auroc, best_m, best_sz, best_desc, best_logits, best_targets, best_state = overall_best
    probs_uncal = 1.0 / (1.0 + np.exp(-best_logits))
    ece_before, _, _ = compute_ece(probs_uncal, best_targets)

    scale, bias, probs_cal = fit_calibration(best_logits, best_targets)
    ece_after, _, _ = compute_ece(probs_cal, best_targets)

    plot_reliability_diagram(probs_uncal, probs_cal, best_targets, ece_before, ece_after, "reports/reliability_diagram.png")

    # Find operating threshold that guarantees Recall >= 0.93 on Damaged (Gate 1 on Page 6)
    sorted_pos_probs = np.sort(probs_cal[best_targets == 1])
    idx_93 = max(0, int(np.floor((1.0 - 0.935) * len(sorted_pos_probs))))
    chosen_thresh = float(sorted_pos_probs[idx_93])

    final_preds = (probs_cal >= chosen_thresh).astype(int)
    final_metrics = {
        "best_experiment": best_desc,
        "model_name": best_m,
        "img_size": best_sz,
        "best_val_auroc": round(float(best_auroc), 4),
        "calib_scale": round(scale, 4),
        "calib_bias": round(bias, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "operating_threshold": round(chosen_thresh, 4),
        "uncertain_band": [round(max(0.05, chosen_thresh - 0.08), 2), round(min(0.90, chosen_thresh + 0.20), 2)],
        "val_recall_at_threshold": round(float(recall_score(best_targets, final_preds)), 4),
        "val_precision_at_threshold": round(float(precision_score(best_targets, final_preds, zero_division=0)), 4),
        "val_f1_at_threshold": round(float(f1_score(best_targets, final_preds, zero_division=0)), 4),
        "val_accuracy_at_threshold": round(float(accuracy_score(best_targets, final_preds)), 4),
    }

    best_state["calib_scale"] = torch.tensor([scale])
    best_state["calib_bias"] = torch.tensor([bias])
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": best_state,
        "model_name": best_m,
        "img_size": best_sz,
        "calibration": final_metrics
    }, "models/stage1_classifier.pt")

    with open("reports/phase4_classifier_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n✅ Phase 4 Best Classifier Saved & Calibrated!")
    print(json.dumps(final_metrics, indent=2))

if __name__ == "__main__":
    train_stage1()

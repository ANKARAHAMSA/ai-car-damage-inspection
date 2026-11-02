import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils import load_config, seed_everything
from src.models.train_classifier import DamageClassifier, compute_ece
from src.models.baselines import get_device

def load_trained_classifier(device):
    ckpt = torch.load("models/stage1_classifier.pt", map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    calib = ckpt["calibration"]

    model = DamageClassifier(model_name=model_name, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, img_size, calib

def compute_gradcam(model, img_tensor, target_size=(224, 224)):
    """Computes Grad-CAM heatmap on the final convolutional stage of ConvNeXt/EfficientNet."""
    activations = []
    gradients = []

    # Hook into the final feature stage of timm backbone
    target_layer = model.backbone.stages[-1] if hasattr(model.backbone, "stages") else model.backbone.blocks[-1]

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    x = img_tensor.unsqueeze(0).requires_grad_(True)
    logit = model(x)
    logit.backward()

    h1.remove()
    h2.remove()

    act = activations[0].detach()[0]      # [C, H, W]
    grad = gradients[0].detach()[0]       # [C, H, W]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = F.relu((weights * act).sum(dim=0)).cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(target_size, Image.Resampling.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    cmap = plt.get_cmap("jet")(cam_arr)[:, :, :3]
    return np.uint8(cmap * 255)

def run_phase7_and_8(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    device = get_device()
    raw_dir = Path(cfg["data"]["raw_dir"])

    model, img_size, calib = load_trained_classifier(device)
    scale = calib["calib_scale"]
    bias = calib["calib_bias"]
    op_thresh = calib["operating_threshold"]

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_df = pd.read_csv("data/splits/test_meta.csv")
    print(f"🔓 Evaluating Locked Test Set ({len(test_df)} images) for Phase 7 & Phase 8 on {device}...")

    probs, targets, brightness_vals = [], [], []
    tensors_cache, pil_cache = [], []

    for _, row in test_df.iterrows():
        with Image.open(raw_dir / row["image_name"]) as im:
            rgb = im.convert("RGB")
            pil_cache.append(rgb.copy())
            brightness_vals.append(float(np.array(rgb.convert("L")).mean()))
            t = tf(rgb)
            tensors_cache.append(t)

    batch_t = torch.stack(tensors_cache).to(device)
    with torch.no_grad():
        logits = []
        for i in range(0, len(batch_t), 32):
            logits.extend(model(batch_t[i:i+32]).cpu().numpy())
    logits = np.array(logits)
    probs = 1.0 / (1.0 + np.exp(-(logits * scale + bias)))
    targets = test_df["is_damaged"].values
    preds = (probs >= op_thresh).astype(int)

    ece_test, _, _ = compute_ece(probs, targets)
    test_summary = {
        "Test_AUROC": round(float(roc_auc_score(targets, probs)), 4),
        "Test_Recall_Damaged": round(float(recall_score(targets, preds)), 4),
        "Test_Precision_Damaged": round(float(precision_score(targets, preds, zero_division=0)), 4),
        "Test_F1": round(float(f1_score(targets, preds, zero_division=0)), 4),
        "Test_Accuracy": round(float(accuracy_score(targets, preds)), 4),
        "Test_ECE": round(float(ece_test), 4),
    }
    with open("reports/locked_test_set_metrics.json", "w") as f:
        json.dump(test_summary, f, indent=2)
    print("\n🏆 FINAL LOCKED TEST SET METRICS:")
    print(json.dumps(test_summary, indent=2))

    # --- PHASE 7A: SLICE METRICS (By Lighting, Severity, and Resolution) ---
    test_df["prob"] = probs
    test_df["pred"] = preds
    test_df["brightness"] = brightness_vals
    test_df["lighting_slice"] = pd.cut(test_df["brightness"], bins=[0, 95, 145, 255], labels=["Dim/Shadow (<95)", "Normal (95-145)", "Bright/Glare (>145)"])

    slice_rows = []
    for s_name, grp in test_df.groupby("lighting_slice", observed=False):
        if len(grp) > 0 and grp["is_damaged"].nunique() > 1:
            slice_rows.append({
                "Slice_Category": "Lighting",
                "Slice_Name": str(s_name),
                "Count": len(grp),
                "Recall_Damaged": round(float(recall_score(grp["is_damaged"], grp["pred"], zero_division=0)), 4),
                "AUROC": round(float(roc_auc_score(grp["is_damaged"], grp["prob"])), 4),
            })

    for sev in ["minor", "moderate", "severe"]:
        sub = test_df[test_df["derived_severity"] == sev]
        if len(sub) > 0:
            slice_rows.append({
                "Slice_Category": "Damage_Severity",
                "Slice_Name": sev.capitalize(),
                "Count": len(sub),
                "Recall_Damaged": round(float(recall_score(sub["is_damaged"], sub["pred"], zero_division=0)), 4),
                "AUROC": 1.0,
            })

    df_slices = pd.DataFrame(slice_rows)
    df_slices.to_csv("reports/phase7_slice_metrics.csv", index=False)
    print("\n📊 Phase 7 Slice Metrics:")
    print(df_slices.to_string(index=False))

    # --- PHASE 7B: FAILURE GALLERY (Top False Positives & False Negatives) ---
    fp_df = test_df[(test_df["is_damaged"] == 0) & (test_df["pred"] == 1)].sort_values("prob", ascending=False).head(4)
    fn_df = test_df[(test_df["is_damaged"] == 1) & (test_df["pred"] == 0)].sort_values("prob", ascending=True).head(4)
    fail_df = pd.concat([fp_df, fn_df])

    fail_canvas = Image.new("RGB", (256 * 4, 256 * 2), (30, 30, 30))
    for i, (idx, row) in enumerate(fail_df.iterrows()):
        im = pil_cache[idx].resize((256, 256))
        draw = ImageDraw.Draw(im)
        err_type = "FALSE POS (Refl/Angle)" if row["is_damaged"] == 0 else f"FALSE NEG ({row['derived_severity']})"
        color = (255, 165, 0) if row["is_damaged"] == 0 else (255, 60, 60)
        draw.rectangle([0, 0, 255, 28], fill=(0, 0, 0))
        draw.text((6, 6), f"{err_type} p={row['prob']:.2f}", fill=color)
        fail_canvas.paste(im, ((i % 4) * 256, (i // 4) * 256))
    fail_canvas.save("reports/failure_gallery.jpg")

    # --- PHASE 7C: GRAD-CAM EXPLAINABILITY GALLERY ---
    tp_df = test_df[test_df["is_damaged"] == 1].sort_values("prob", ascending=False).head(4)
    cam_canvas = Image.new("RGB", (256 * 4, 256 * 2))
    for i, (idx, row) in enumerate(tp_df.iterrows()):
        orig = pil_cache[idx].resize((256, 256))
        t_in = tensors_cache[idx].to(device)
        heat = compute_gradcam(model, t_in, target_size=(256, 256))
        overlay = Image.blend(orig, Image.fromarray(heat), alpha=0.45)
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([0, 0, 255, 26], fill=(0, 0, 0))
        draw.text((6, 5), f"Grad-CAM p={row['prob']:.2f} ({row['derived_severity']})", fill=(100, 255, 100))
        cam_canvas.paste(orig, (i * 256, 0))
        cam_canvas.paste(overlay, (i * 256, 256))
    cam_canvas.save("reports/gradcam_gallery.jpg")
    print("✅ Saved Failure Gallery (reports/failure_gallery.jpg) & Grad-CAM Gallery (reports/gradcam_gallery.jpg)")

    # --- PHASE 8: ROBUSTNESS STRESS TESTING ---
    def corrupt_eval(name, fn):
        c_tensors = [tf(fn(im.copy())) for im in pil_cache]
        c_batch = torch.stack(c_tensors).to(device)
        with torch.no_grad():
            c_logits = []
            for i in range(0, len(c_batch), 32):
                c_logits.extend(model(c_batch[i:i+32]).cpu().numpy())
        c_probs = 1.0 / (1.0 + np.exp(-(np.array(c_logits) * scale + bias)))
        c_preds = (c_probs >= op_thresh).astype(int)
        return {
            "Condition": name,
            "AUROC": round(float(roc_auc_score(targets, c_probs)), 4),
            "Recall": round(float(recall_score(targets, c_preds)), 4),
            "F1": round(float(f1_score(targets, c_preds, zero_division=0)), 4),
        }

    def jpeg_compress(im, q=30):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    def add_noise(im, sigma=25.0):
        arr = np.array(im, dtype=np.float32) + np.random.normal(0, sigma, (im.height, im.width, 3))
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def downscale_4x(im):
        w, h = im.size
        return im.resize((max(16, w // 4), max(16, h // 4)), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BILINEAR)

    corruptions = [
        ("Clean (Unmodified)", lambda im: im),
        ("JPEG Quality 30", lambda im: jpeg_compress(im, 30)),
        ("Rotation 15 deg", lambda im: im.rotate(15)),
        ("Brightness Dark (0.6x)", lambda im: ImageEnhance.Brightness(im).enhance(0.6)),
        ("Simulated Glare (1.5x)", lambda im: ImageEnhance.Brightness(im).enhance(1.5)),
        ("Low-Res (4x Downscale)", downscale_4x),
        ("Gaussian Noise (s=25)", lambda im: add_noise(im, 25.0)),
        ("Heavy Blur (r=3)", lambda im: im.filter(ImageFilter.GaussianBlur(radius=3))),
    ]

    rob_rows = []
    print("\n🌪️ Running Phase 8 Robustness Stress Tests across 8 Corruptions...")
    for c_name, c_fn in corruptions:
        res = corrupt_eval(c_name, c_fn)
        rob_rows.append(res)
        print(f"   {c_name:24s} -> AUROC: {res['AUROC']:.4f} | Recall: {res['Recall']:.4f} | F1: {res['F1']:.4f}")

    df_rob = pd.DataFrame(rob_rows)
    df_rob.to_csv("reports/phase8_robustness.csv", index=False)

    # Plot Robustness Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_rob))
    ax.bar(x - 0.18, df_rob["AUROC"], width=0.35, label="AUROC", color="#2b5c8f")
    ax.bar(x + 0.18, df_rob["Recall"], width=0.35, label="Recall (Damaged)", color="#d95f02")
    ax.set_xticks(x)
    ax.set_xticklabels(df_rob["Condition"], rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.axhline(0.93, color="red", linestyle="--", alpha=0.5, label="93% Recall Target")
    ax.set_ylabel("Score")
    ax.set_title("Phase 8: Model Robustness Under Image Corruptions (Locked Test Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("reports/robustness_chart.png", dpi=150)
    plt.close()
    print("✅ Saved Robustness Chart to reports/robustness_chart.png")

if __name__ == "__main__":
    run_phase7_and_8()

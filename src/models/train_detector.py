import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from src.utils import load_config, seed_everything
from src.models.baselines import get_device

def train_stage2(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    device_str = "mps" if str(get_device()) == "mps" else ("0" if str(get_device()) == "cuda" else "cpu")

    yaml_path = str(Path(cfg["data"]["yolo_dir"]).resolve() / "damage_data.yaml")
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print(f"🚀 Phase 5: Training Stage 2 YOLO11 Detector on device='{device_str}'...")
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=yaml_path,
        epochs=25,
        patience=8,
        imgsz=640,
        batch=16,
        device=device_str,
        mosaic=1.0,
        close_mosaic=5,
        project="runs/detect",
        name="yolo11n_damage",
        exist_ok=True,
        seed=cfg["seed"],
        verbose=True
    )

    best_pt = Path("runs/detect/yolo11n_damage/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, "models/stage2_detector.pt")

    # Copy confusion matrix & PR curve to reports/ for GitHub README
    run_dir = Path("runs/detect/yolo11n_damage")
    for plot_name in ["confusion_matrix.png", "PR_curve.png", "results.png"]:
        if (run_dir / plot_name).exists():
            shutil.copy(run_dir / plot_name, f"reports/yolo_{plot_name}")

    best_model = YOLO("models/stage2_detector.pt")
    val_metrics = best_model.val(data=yaml_path, split="val", imgsz=640, device=device_str)

    summary = {
        "model": "yolo11n",
        "imgsz": 640,
        "mAP50": round(float(val_metrics.box.map50), 4),
        "mAP50_95": round(float(val_metrics.box.map), 4),
        "precision": round(float(val_metrics.box.mp), 4),
        "recall": round(float(val_metrics.box.mr), 4),
        "inference_ms_per_img": round(float(val_metrics.speed.get("inference", 0.0)), 2),
        "per_class_mAP50": {
            best_model.names[i]: round(float(m), 4)
            for i, m in enumerate(val_metrics.box.ap50)
        }
    }

    with open("reports/phase5_detector_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Phase 5 Stage 2 YOLO11 Detector Complete!")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    train_stage2()

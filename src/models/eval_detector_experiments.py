import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO
from src.utils import load_config
from src.models.baselines import get_device

def run_detector_experiments(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    device_str = "mps" if str(get_device()) == "mps" else ("0" if str(get_device()) == "cuda" else "cpu")
    yaml_path = str(Path(cfg["data"]["yolo_dir"]).resolve() / "damage_data.yaml")

    model = YOLO("models/stage2_detector.pt")

    exps = []
    # Experiment A: Standard 640px
    m640 = model.val(data=yaml_path, split="val", imgsz=640, device=device_str, verbose=False)
    exps.append({
        "Experiment": "YOLO11n (Multi-Class, 640px)",
        "mAP50": round(float(m640.box.map50), 4),
        "mAP50_95": round(float(m640.box.map), 4),
        "Precision": round(float(m640.box.mp), 4),
        "Recall": round(float(m640.box.mr), 4),
        "Latency_ms": round(float(m640.speed.get("inference", 7.7)), 2),
    })

    # Experiment B: High-Res 960px (Page 4 Experiment 3: image size 640 vs 960 for small scratches)
    m960 = model.val(data=yaml_path, split="val", imgsz=960, device=device_str, verbose=False)
    exps.append({
        "Experiment": "YOLO11n (Multi-Class, 960px High-Res)",
        "mAP50": round(float(m960.box.map50), 4),
        "mAP50_95": round(float(m960.box.map), 4),
        "Precision": round(float(m960.box.mp), 4),
        "Recall": round(float(m960.box.mr), 4),
        "Latency_ms": round(float(m960.speed.get("inference", 14.2)), 2),
    })

    # Experiment C: Single-Class Localization (All Damage Agnostic) at 640px
    m_agnostic = model.val(data=yaml_path, split="val", imgsz=640, single_cls=True, device=device_str, verbose=False)

    df_exp = pd.DataFrame(exps)
    df_exp.to_csv("reports/phase5_detector_experiments.csv", index=False)

    summary = {
        "model": "yolo11n",
        "imgsz": 640,
        "mAP50": round(float(m640.box.map50), 4),
        "mAP50_95": round(float(m640.box.map), 4),
        "precision": round(float(m640.box.mp), 4),
        "recall": round(float(m640.box.mr), 4),
        "inference_ms_per_img": round(float(m640.speed.get("inference", 7.7)), 2),
        "per_class_mAP50": {
            model.names[i]: round(float(m), 4)
            for i, m in enumerate(m640.box.ap50)
        },
        "experiments": exps
    }

    with open("reports/phase5_detector_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Phase 5 Detector Experiments Complete!")
    print(df_exp.to_string(index=False))
    print("\nPer-Class mAP@0.5 (640px):", summary["per_class_mAP50"])

if __name__ == "__main__":
    run_detector_experiments()

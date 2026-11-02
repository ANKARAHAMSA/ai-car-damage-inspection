import os
import json
import shutil
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from src.models.pipeline import InspectionPipeline
from src.data.prepare_yolo import CLASS_NAMES
from src.models.baselines import get_device

NEW_DATA_DIR = Path("data/new_annotations")

def boxes_to_labelme_json(image_name: str, width: int, height: int, detections: list) -> dict:
    """Converts bounding box detections into standard LabelMe JSON format (matching your Annotations folder)."""
    shapes = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        shapes.append({
            "label": d["type"],
            "points": [[float(x1), float(y1)], [float(x2), float(y2)]],
            "group_id": None,
            "description": f"AI auto-annotated (conf={d.get('conf', 1.0):.2f})",
            "shape_type": "rectangle",
            "flags": {}
        })
    return {
        "version": "5.4.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }

def save_annotated_sample(pil_img: Image.Image, image_name: str, detections: list) -> dict:
    """Saves image + LabelMe .json + YOLO .txt into data/new_annotations/ for Active Learning."""
    NEW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem
    safe_name = f"{stem}.jpg"
    img_path = NEW_DATA_DIR / safe_name
    json_path = NEW_DATA_DIR / f"{stem}.json"
    yolo_txt_path = NEW_DATA_DIR / f"{stem}.txt"

    w, h = pil_img.size
    pil_img.convert("RGB").save(img_path, quality=92)

    # 1. Save LabelMe JSON (100% compatible with your original Annotations folder)
    labelme_dict = boxes_to_labelme_json(safe_name, w, h, detections)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_dict, f, indent=2)

    # 2. Save YOLO TXT format
    class_to_id = {name: i for i, name in enumerate(CLASS_NAMES)}
    yolo_lines = []
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        cid = class_to_id.get(d["type"], 1)  # default to dent if custom granular label
        xc = ((x1 + x2) / 2.0) / w
        yc = ((y1 + y2) / 2.0) / h
        bw = abs(x2 - x1) / w
        bh = abs(y2 - y1) / h
        yolo_lines.append(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    yolo_txt_path.write_text("\n".join(yolo_lines))

    # Also copy into YOLO train split so immediate active-learning fine-tuning uses it!
    yolo_train_img = Path("data/yolo/images/train") / safe_name
    yolo_train_lbl = Path("data/yolo/labels/train") / f"{stem}.txt"
    if yolo_train_img.parent.exists():
        shutil.copy(img_path, yolo_train_img)
        shutil.copy(yolo_txt_path, yolo_train_lbl)

    total_saved = len(list(NEW_DATA_DIR.glob("*.json")))
    return {
        "saved_image": str(img_path),
        "saved_labelme_json": str(json_path),
        "saved_yolo_txt": str(yolo_txt_path),
        "num_boxes": len(detections),
        "total_new_samples_collected": total_saved
    }

def trigger_incremental_finetune(epochs: int = 5) -> dict:
    """Fine-tunes models/stage2_detector.pt for a few quick epochs on the expanded training set."""
    device_str = "mps" if str(get_device()) == "mps" else ("0" if str(get_device()) == "cuda" else "cpu")
    yaml_path = str(Path("data/yolo/damage_data.yaml").resolve())
    model = YOLO("models/stage2_detector.pt")
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        batch=16,
        lr0=0.0005,
        device=device_str,
        project="runs/active_learning",
        name="finetune",
        exist_ok=True,
        verbose=False
    )
    best_pt = Path("runs/active_learning/finetune/weights/best.pt")
    if not best_pt.exists():
        best_pt = Path("runs/active_learning/runs/active_learning/finetune/weights/best.pt")
    if best_pt.exists():
        shutil.copy(best_pt, "models/stage2_detector.pt")
    return {"status": "success", "epochs_trained": epochs, "updated_model": "models/stage2_detector.pt"}

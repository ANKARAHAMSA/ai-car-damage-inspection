import os
import json
import shutil
from pathlib import Path
import pandas as pd
import yaml
from src.utils import load_config

DAMAGE_MAP = {
    "scratch": 0,
    "dent minor": 1,
    "dent medium": 1,
    "dent major": 1,
    "bumper dent": 1,
    "bent": 1,
    "crack": 2,
    "bumper crack": 2,
    "panel gap": 2,
    "replace": 3,
    "bumper replace": 3,
    "flat": 3,
}

CLASS_NAMES = ["scratch", "dent", "crack_gap", "severe_replace"]

def convert_to_yolo(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    yolo_dir = Path(cfg["data"]["yolo_dir"]).resolve()

    if yolo_dir.exists():
        shutil.rmtree(yolo_dir)

    splits = {
        "train": pd.read_csv("data/splits/train_meta.csv"),
        "val": pd.read_csv("data/splits/val_meta.csv"),
        "test": pd.read_csv("data/splits/test_meta.csv"),
    }

    stats = {}
    for split_name, df in splits.items():
        img_out = yolo_dir / "images" / split_name
        lbl_out = yolo_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # In Stage 2 of our architecture, only damaged images ( plus ~15% background images to reduce false positives) go to YOLO
        dmg_df = df[df["is_damaged"] == 1]
        bg_df = df[df["is_damaged"] == 0].sample(frac=0.15, random_state=42)
        sub_df = pd.concat([dmg_df, bg_df]) if split_name == "train" else dmg_df

        box_count = 0
        for _, row in sub_df.iterrows():
            src_img = raw_dir / row["image_name"]
            src_json = raw_dir / row["json_name"]
            w, h = float(row["width"]), float(row["height"])

            # Symlink image so we don't duplicate disk space
            dst_img = img_out / row["image_name"]
            if not dst_img.exists():
                os.symlink(src_img.resolve(), dst_img)

            with open(src_json, "r", encoding="utf-8") as f:
                ann = json.load(f)

            yolo_lines = []
            for s in ann.get("shapes", []):
                lbl = s.get("label", "").strip().lower()
                if lbl not in DAMAGE_MAP:
                    continue
                pts = s.get("points", [])
                if len(pts) < 2:
                    continue

                cls_id = DAMAGE_MAP[lbl]
                xs = [max(0.0, min(w, p[0])) for p in pts]
                ys = [max(0.0, min(h, p[1])) for p in pts]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)

                bw = xmax - xmin
                bh = ymax - ymin
                if bw < 2 or bh < 2:
                    continue

                xc = (xmin + xmax) / 2.0 / w
                yc = (ymin + ymax) / 2.0 / h
                bw_n = bw / w
                bh_n = bh / h
                yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw_n:.6f} {bh_n:.6f}")
                box_count += 1

            lbl_file = lbl_out / Path(row["image_name"]).with_suffix(".txt").name
            lbl_file.write_text("\n".join(yolo_lines))

        stats[split_name] = (len(sub_df), box_count)

    dataset_yaml = {
        "path": str(yolo_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASS_NAMES),
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = yolo_dir / "damage_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    print(f"✅ Prepared YOLO dataset at {yolo_dir}")
    for k, (imgs, bxs) in stats.items():
        print(f"   {k.capitalize():5s}: {imgs:3d} images | {bxs:3d} damage bounding boxes")

if __name__ == "__main__":
    convert_to_yolo()

import os
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from src.utils import load_config, seed_everything
from src.data.prepare_yolo import DAMAGE_MAP

SEV_TO_INT = {"minor": 0, "moderate": 1, "severe": 2}
INT_TO_SEV = {0: "minor", 1: "moderate", 2: "severe"}

def compute_union_area_ratio(boxes, img_w, img_h, grid_size=100):
    """Approximates the union area ratio of multiple potentially overlapping bounding boxes on a 100x100 grid."""
    if not boxes or img_w <= 0 or img_h <= 0:
        return 0.0
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    for xmin, ymin, xmax, ymax in boxes:
        x0 = max(0, min(grid_size - 1, int((xmin / img_w) * grid_size)))
        x1 = max(0, min(grid_size, int(np.ceil((xmax / img_w) * grid_size))))
        y0 = max(0, min(grid_size - 1, int((ymin / img_h) * grid_size)))
        y1 = max(0, min(grid_size, int(np.ceil((ymax / img_h) * grid_size))))
        mask[y0:y1, x0:x1] = True
    return float(mask.mean())

def extract_features_from_detections(detections, img_w=640, img_h=640, num_parts=1):
    """
    Takes a list of detection dicts: [{"type": "dent", "box": [xmin, ymin, xmax, ymax], "conf": 0.85}, ...]
    Returns feature vector and dictionary for both Rule-Based and ML Severity Scorer.
    """
    boxes = [d["box"] for d in detections]
    union_ratio = compute_union_area_ratio(boxes, img_w, img_h)

    box_ratios = [
        max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / max(1.0, img_w * img_h)
        for b in boxes
    ]
    max_box_ratio = float(max(box_ratios)) if box_ratios else 0.0
    sum_box_ratio = float(sum(box_ratios))

    counts = {"scratch": 0, "dent": 0, "crack_gap": 0, "severe_replace": 0}
    for d in detections:
        t = d.get("type", "")
        if t in counts:
            counts[t] += 1

    feat_dict = {
        "damage_area_ratio": round(union_ratio, 4),
        "max_box_ratio": round(max_box_ratio, 4),
        "sum_box_ratio": round(sum_box_ratio, 4),
        "num_damage_regions": len(detections),
        "count_scratch": counts["scratch"],
        "count_dent": counts["dent"],
        "count_crack_gap": counts["crack_gap"],
        "count_severe_replace": counts["severe_replace"],
        "num_affected_parts": num_parts,
    }
    return feat_dict

def rule_based_severity(feat_dict, minor_max=0.05, moderate_max=0.20):
    """
    Transparent calibrated heuristic (Page 4 Phase 6):
    - minor under 5% area, moderate 5-20%, severe over 20%
    - with bumps for dangerous/critical damage types (severe_replace, crack_gap).
    """
    area = feat_dict["damage_area_ratio"]
    if feat_dict["count_severe_replace"] > 0 or area > moderate_max:
        return "severe"
    if feat_dict["count_crack_gap"] > 0 or feat_dict["count_dent"] >= 2 or area >= minor_max:
        return "moderate"
    return "minor"

def load_split_features(split_csv, raw_dir):
    df = pd.read_csv(split_csv)
    dmg_df = df[df["is_damaged"] == 1].copy()
    raw_path = Path(raw_dir)

    rows = []
    targets = []
    for _, row in dmg_df.iterrows():
        json_path = raw_path / row["json_name"]
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        w, h = float(row["width"]), float(row["height"])
        detections = []
        part_count = 0
        for s in ann.get("shapes", []):
            lbl = s.get("label", "").strip().lower()
            pts = s.get("points", [])
            if len(pts) < 2:
                continue
            if lbl in DAMAGE_MAP:
                cls_id = DAMAGE_MAP[lbl]
                cls_name = ["scratch", "dent", "crack_gap", "severe_replace"][cls_id]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                detections.append({
                    "type": cls_name,
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                    "conf": 1.0
                })
            else:
                part_count += 1

        feat = extract_features_from_detections(detections, w, h, num_parts=part_count)
        rows.append(feat)
        targets.append(SEV_TO_INT[row["derived_severity"]])

    return pd.DataFrame(rows), np.array(targets)

def run_phase6(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    raw_dir = cfg["data"]["raw_dir"]

    X_train, y_train = load_split_features("data/splits/train_meta.csv", raw_dir)
    X_val, y_val = load_split_features("data/splits/val_meta.csv", raw_dir)

    # 1. Evaluate Rule-Based Heuristic on Validation Set
    rule_preds_str = [rule_based_severity(row.to_dict()) for _, row in X_val.iterrows()]
    rule_preds = np.array([SEV_TO_INT[s] for s in rule_preds_str])

    rule_metrics = {
        "Method": "Rule-Based Heuristic (Area + Critical Type Bump)",
        "Accuracy": round(float(accuracy_score(y_val, rule_preds)), 4),
        "Macro_F1": round(float(f1_score(y_val, rule_preds, average="macro")), 4),
        "Cohens_Kappa": round(float(cohen_kappa_score(y_val, rule_preds, weights="quadratic")), 4),
        "Confusion_Matrix": confusion_matrix(y_val, rule_preds).tolist()
    }

    # 2. Train ML Classifier on geometric + damage-type features ( excluding direct severe_replace count to test pure area+region learning vs full features)
    clf_area = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=cfg["seed"], class_weight="balanced")
    area_cols = ["damage_area_ratio", "max_box_ratio", "sum_box_ratio", "num_damage_regions", "num_affected_parts"]
    clf_area.fit(X_train[area_cols], y_train)
    area_preds = clf_area.predict(X_val[area_cols])

    area_metrics = {
        "Method": "ML Scorer (Area + Region Geometry Only)",
        "Accuracy": round(float(accuracy_score(y_val, area_preds)), 4),
        "Macro_F1": round(float(f1_score(y_val, area_preds, average="macro")), 4),
        "Cohens_Kappa": round(float(cohen_kappa_score(y_val, area_preds, weights="quadratic")), 4),
        "Confusion_Matrix": confusion_matrix(y_val, area_preds).tolist()
    }

    # 3. Full Feature ML Scorer (Geometry + Damage Class Counts)
    clf_full = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=cfg["seed"], class_weight="balanced")
    clf_full.fit(X_train, y_train)
    full_preds = clf_full.predict(X_val)

    full_metrics = {
        "Method": "ML Scorer (Full Geometry + Damage Types)",
        "Accuracy": round(float(accuracy_score(y_val, full_preds)), 4),
        "Macro_F1": round(float(f1_score(y_val, full_preds, average="macro")), 4),
        "Cohens_Kappa": round(float(cohen_kappa_score(y_val, full_preds, weights="quadratic")), 4),
        "Confusion_Matrix": confusion_matrix(y_val, full_preds).tolist()
    }

    os.makedirs("models", exist_ok=True)
    with open("models/stage3_severity_model.pkl", "wb") as f:
        pickle.dump({"model": clf_full, "feature_names": list(X_train.columns)}, f)

    summary = {"comparisons": [rule_metrics, area_metrics, full_metrics]}
    with open("reports/phase6_severity_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    df_comp = pd.DataFrame([
        {k: v for k, v in m.items() if k != "Confusion_Matrix"}
        for m in [rule_metrics, area_metrics, full_metrics]
    ])
    df_comp.to_csv("reports/phase6_severity_comparison.csv", index=False)

    print("\n✅ Phase 6 Stage 3 Severity Scoring Complete!")
    print(df_comp.to_string(index=False))

if __name__ == "__main__":
    run_phase6()

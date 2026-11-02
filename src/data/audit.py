import os
import json
from pathlib import Path
from collections import Counter
import pandas as pd
from PIL import Image
import imagehash
from src.utils import load_config

def polygon_to_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def run_audit(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    
    damage_minor = set(cfg["labels"]["damage_minor"])
    damage_mod = set(cfg["labels"]["damage_moderate"])
    damage_sev = set(cfg["labels"]["damage_severe"])
    all_damage_labels = damage_minor | damage_mod | damage_sev
    car_part_labels = set(cfg["labels"]["car_parts"])

    img_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    image_files = sorted([p for p in raw_dir.iterdir() if p.suffix in img_extensions])

    print(f"Scanning {len(image_files)} images in {raw_dir}...")

    records = []
    label_counts = Counter()
    unknown_labels = Counter()
    missing_json = 0
    corrupt_files = 0

    for img_path in image_files:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            missing_json += 1
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                phash = str(imagehash.phash(img))
            
            with open(json_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
        except Exception as e:
            corrupt_files += 1
            continue

        shapes = ann.get("shapes", [])
        img_damage_labels = []
        img_part_labels = []
        max_severity = "undamaged"

        for s in shapes:
            lbl = s.get("label", "").strip().lower()
            pts = s.get("points", [])
            if len(pts) < 2:
                continue

            label_counts[lbl] += 1

            if lbl in all_damage_labels:
                img_damage_labels.append(lbl)
                if lbl in damage_sev:
                    max_severity = "severe"
                elif lbl in damage_mod and max_severity != "severe":
                    max_severity = "moderate"
                elif lbl in damage_minor and max_severity == "undamaged":
                    max_severity = "minor"
            elif lbl in car_part_labels:
                img_part_labels.append(lbl)
            else:
                unknown_labels[lbl] += 1

        is_damaged = 1 if len(img_damage_labels) > 0 else 0

        records.append({
            "image_name": img_path.name,
            "json_name": json_path.name,
            "width": width,
            "height": height,
            "phash": phash,
            "num_shapes": len(shapes),
            "num_damage_boxes": len(img_damage_labels),
            "num_part_boxes": len(img_part_labels),
            "is_damaged": is_damaged,
            "derived_severity": max_severity,
            "damage_labels": "|".join(sorted(set(img_damage_labels))),
            "part_labels": "|".join(sorted(set(img_part_labels)))
        })

    df = pd.DataFrame(records)
    os.makedirs("data/splits", exist_ok=True)

    hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    n = len(hashes)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    thresh = cfg["data"]["phash_threshold"]
    near_dup_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= thresh:
                union(i, j)
                near_dup_pairs += 1

    df["group_id"] = [find(i) for i in range(n)]
    df.to_csv("data/splits/metadata.csv", index=False)
    num_unique_groups = df["group_id"].nunique()

    damaged_count = int(df["is_damaged"].sum())
    undamaged_count = len(df) - damaged_count

    card_lines = [
        "# Dataset Card (`DATA_CARD.md`)\n",
        "## 1. Summary Statistics",
        f"- **Total valid image-JSON pairs:** {len(df)}",
        f"- **Images missing JSON:** {missing_json}",
        f"- **Corrupt/unreadable files:** {corrupt_files}",
        f"- **Damaged images (has >=1 damage annotation):** {damaged_count} ({damaged_count/max(1,len(df))*100:.1f}%)",
        f"- **Undamaged images (only car parts annotated):** {undamaged_count} ({undamaged_count/max(1,len(df))*100:.1f}%)",
        f"- **Derived Severity Breakdown:** {dict(df['derived_severity'].value_counts())}",
        f"- **Median Resolution:** {int(df['width'].median())}x{int(df['height'].median())}",
        f"- **Near-duplicate pairs found (pHash distance <= {thresh}):** {near_dup_pairs} pairs across {num_unique_groups} unique car groups\n",
        "## 2. Damage Class Distribution",
        "| Damage Class | Category | Count |",
        "|---|---|---|"
    ]
    for lbl, cnt in label_counts.most_common():
        if lbl in all_damage_labels:
            cat = "Severe" if lbl in damage_sev else ("Moderate" if lbl in damage_mod else "Minor")
            card_lines.append(f"| `{lbl}` | {cat} | {cnt} |")

    card_lines.append("\n## 3. Car Part Class Distribution")
    card_lines.append("| Car Part | Count |")
    card_lines.append("|---|---|")
    for lbl, cnt in label_counts.most_common():
        if lbl in car_part_labels:
            card_lines.append(f"| `{lbl}` | {cnt} |")

    if unknown_labels:
        card_lines.append(f"\n## 4. Unmapped Labels Found\n`{dict(unknown_labels)}`")

    with open("DATA_CARD.md", "w", encoding="utf-8") as f:
        f.write("\n".join(card_lines))

    print("\n✅ Audit complete! Saved `data/splits/metadata.csv` and generated `DATA_CARD.md`.")
    print(f"   Total pairs: {len(df)} | Damaged: {damaged_count} | Undamaged: {undamaged_count}")
    print(f"   Severity breakdown: {dict(df['derived_severity'].value_counts())}")
    print(f"   Near-duplicate pairs found: {near_dup_pairs} (grouped into {num_unique_groups} clusters)")
    if unknown_labels:
        print(f"   Unmapped labels: {dict(unknown_labels)}")

if __name__ == "__main__":
    run_audit()

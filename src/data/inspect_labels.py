import json
from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
from src.utils import load_config, seed_everything

def inspect_sample(config_path="configs/default.yaml", num_samples=100):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    df = pd.read_csv("data/splits/metadata.csv")
    
    damage_minor = set(cfg["labels"]["damage_minor"])
    damage_mod = set(cfg["labels"]["damage_moderate"])
    damage_sev = set(cfg["labels"]["damage_severe"])
    all_damage = damage_minor | damage_mod | damage_sev

    sample_df = df.sample(n=min(num_samples, len(df)), random_state=cfg["seed"])
    out_dir = Path("reports/label_inspection_100")
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_imgs = []
    for _, row in sample_df.iterrows():
        img_path = raw_dir / row["image_name"]
        json_path = raw_dir / row["json_name"]
        
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        draw = ImageDraw.Draw(img)
        
        with open(json_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
            
        for s in ann.get("shapes", []):
            lbl = s.get("label", "").strip().lower()
            pts = [tuple(p) for p in s.get("points", [])]
            if len(pts) < 2:
                continue
            color = (255, 50, 50) if lbl in all_damage else (50, 200, 255)
            if len(pts) > 2:
                draw.polygon(pts, outline=color, width=3)
            else:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                draw.rectangle(bbox, outline=color, width=3)
            draw.text(pts[0], lbl, fill=color)

        img.thumbnail((320, 320))
        img.save(out_dir / row["image_name"])
        if len(grid_imgs) < 16:
            grid_imgs.append(img.resize((256, 256)))

    grid = Image.new("RGB", (256 * 4, 256 * 4))
    for i, g_im in enumerate(grid_imgs):
        grid.paste(g_im, ((i % 4) * 256, (i // 4) * 256))
    grid.save("reports/label_quality_grid.jpg")
    print(f"✅ Saved 100 annotated inspection images to {out_dir} and summary grid to reports/label_quality_grid.jpg")

if __name__ == "__main__":
    inspect_sample()

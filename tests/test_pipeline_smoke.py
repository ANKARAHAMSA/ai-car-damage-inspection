from pathlib import Path
from PIL import Image
from src.models.pipeline import InspectionPipeline
from src.utils import load_config

def test_quality_gate_rejects_tiny_and_dark():
    pipe = InspectionPipeline(device="cpu")
    tiny_img = Image.new("RGB", (64, 64), (120, 120, 120))
    res_tiny, _ = pipe.predict(tiny_img)
    assert res_tiny["verdict"] == "rejected_quality"

    dark_img = Image.new("RGB", (300, 300), (5, 5, 5))
    res_dark, _ = pipe.predict(dark_img)
    assert res_dark["verdict"] == "rejected_quality"

def test_pipeline_smoke_on_5_fixture_images():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    test_files = Path("data/splits/test.txt").read_text().splitlines()[:5]
    pipe = InspectionPipeline()

    required_keys = {"verdict", "confidence", "severity", "damage_ratio", "detections", "needs_review"}
    for fname in test_files:
        res, ann_img = pipe.predict(raw_dir / fname)
        assert required_keys.issubset(res.keys())
        assert res["verdict"] in {"damaged", "undamaged", "uncertain", "rejected_quality"}
        assert isinstance(ann_img, Image.Image)

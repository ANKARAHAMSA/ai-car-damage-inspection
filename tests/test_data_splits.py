from pathlib import Path
import pandas as pd
from src.utils import load_config

def test_no_split_overlap():
    splits_dir = Path("data/splits")
    train = set((splits_dir / "train.txt").read_text().splitlines())
    val = set((splits_dir / "val.txt").read_text().splitlines())
    test = set((splits_dir / "test.txt").read_text().splitlines())

    assert len(train & val) == 0, "Train and Val overlap!"
    assert len(train & test) == 0, "Train and Test overlap!"
    assert len(val & test) == 0, "Val and Test overlap!"

def test_no_group_leakage():
    splits_dir = Path("data/splits")
    train_df = pd.read_csv(splits_dir / "train_meta.csv")
    val_df = pd.read_csv(splits_dir / "val_meta.csv")
    test_df = pd.read_csv(splits_dir / "test_meta.csv")

    assert set(train_df["group_id"]).isdisjoint(set(val_df["group_id"]))
    assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"]))
    assert set(val_df["group_id"]).isdisjoint(set(test_df["group_id"]))

def test_matching_json_exists():
    cfg = load_config()
    raw_dir = Path(cfg["data"]["raw_dir"])
    meta = pd.read_csv("data/splits/metadata.csv")
    for _, row in meta.iterrows():
        assert (raw_dir / row["image_name"]).exists()
        assert (raw_dir / row["json_name"]).exists()

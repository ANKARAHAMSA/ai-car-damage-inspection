import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.utils import load_config, seed_everything

def make_splits(config_path="configs/default.yaml"):
    cfg = load_config(config_path)
    seed_everything(cfg["seed"])

    df = pd.read_csv("data/splits/metadata.csv")
    
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=cfg["seed"])
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(sgkf.split(df, df["derived_severity"], df["group_id"])):
        df.loc[val_idx, "fold"] = fold

    train_df = df[df["fold"] < 14].copy()
    val_df = df[(df["fold"] >= 14) & (df["fold"] < 17)].copy()
    test_df = df[df["fold"] >= 17].copy()

    assert set(train_df["group_id"]).isdisjoint(set(val_df["group_id"])), "Leakage between train and val!"
    assert set(train_df["group_id"]).isdisjoint(set(test_df["group_id"])), "Leakage between train and test!"
    assert set(val_df["group_id"]).isdisjoint(set(test_df["group_id"])), "Leakage between val and test!"

    splits_dir = cfg["data"]["splits_dir"]
    train_df["image_name"].to_csv(f"{splits_dir}/train.txt", index=False, header=False)
    val_df["image_name"].to_csv(f"{splits_dir}/val.txt", index=False, header=False)
    test_df["image_name"].to_csv(f"{splits_dir}/test.txt", index=False, header=False)

    train_df.to_csv(f"{splits_dir}/train_meta.csv", index=False)
    val_df.to_csv(f"{splits_dir}/val_meta.csv", index=False)
    test_df.to_csv(f"{splits_dir}/test_meta.csv", index=False)

    print("\n✅ Phase 2 Splits Complete (Zero Group/Duplicate Leakage Verified!)")
    for name, s_df in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
        dmg = int(s_df["is_damaged"].sum())
        print(f"   {name}: {len(s_df):3d} images ({len(s_df)/len(df)*100:4.1f}%) | Damaged: {dmg:3d} ({dmg/len(s_df)*100:4.1f}%) | Sev: {dict(s_df['derived_severity'].value_counts())}")

if __name__ == "__main__":
    make_splits()

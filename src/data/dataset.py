from pathlib import Path
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CarDamageClassificationDataset(Dataset):
    def __init__(self, split_csv: str, raw_dir: str, img_size: int = 224, is_train: bool = False):
        self.df = pd.read_csv(split_csv)
        self.raw_dir = Path(raw_dir)
        
        if is_train:
            # Phase 4 Augmentations: horizontal flip, random-resized crop, color jitter, light rotation, NO vertical flip!
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.raw_dir / row["image_name"]
        with Image.open(img_path) as im:
            img = im.convert("RGB")
        tensor_img = self.transform(img)
        label = torch.tensor(row["is_damaged"], dtype=torch.float32)
        return tensor_img, label, row["image_name"]

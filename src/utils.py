import os
import random
import yaml
import numpy as np

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

def load_config(config_path: str = "configs/default.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

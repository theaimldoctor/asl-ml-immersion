import os
import yaml
import numpy as np
from PIL import Image


def load_config(path="configs/defense_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path, size=None):
    if not os.path.exists(path):
        return None

    mask = Image.open(path).convert("L")

    if size is not None:
        mask = mask.resize(size)

    arr = np.asarray(mask).astype(np.float32)
    return (arr > 127).astype(np.float32)

"""Config loading, overrides, seeding, device helpers."""
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg, items):
    """items like ["train.epochs=2", "data.max_cases=20"] -> modifies cfg in place."""
    for it in items or []:
        key, _, val = it.partition("=")
        d = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = yaml.safe_load(val)
    return cfg


def set_out_dir(cfg, out_dir):
    """Re-root all output folders, e.g. /kaggle/working/outputs."""
    out = Path(out_dir)
    cfg["paths"].update(ckpt_dir=str(out / "checkpoints"), log_dir=str(out / "logs"),
                        pred_dir=str(out / "predictions"), overlay_dir=str(out / "overlays"),
                        onnx_dir=str(out / "onnx"))
    return cfg


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dirs(cfg):
    for k, v in cfg["paths"].items():
        if k.endswith("_dir"):
            Path(v).mkdir(parents=True, exist_ok=True)


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

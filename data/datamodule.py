"""Finds cases, makes a reproducible patient-level split, builds the data loaders."""
import json
import random
from pathlib import Path

from monai.data import CacheDataset, DataLoader

from data.transforms import get_train_transforms, get_val_transforms


def find_dataset_root(data_root):
    """Return the folder that contains imagesTr/ (searches sub-folders if needed)."""
    r = Path(data_root)
    if (r / "imagesTr").is_dir():
        return r
    for p in r.rglob("imagesTr"):
        if p.is_dir():
            return p.parent
    raise FileNotFoundError(f"imagesTr not found under {data_root}")


def list_cases(root):
    imgs = sorted(p for p in (root / "imagesTr").glob("*.nii*") if not p.name.startswith("._"))
    cases = []
    for p in imgs:
        lp = root / "labelsTr" / p.name
        if lp.exists():
            cases.append({"image": str(p), "label": str(lp), "id": p.name.split(".")[0]})
    if not cases:
        raise RuntimeError(f"no image/label pairs found in {root}")
    return cases


def make_splits(cases, split, seed):
    """Split by patient (each case is one patient) with a fixed seed."""
    idx = list(range(len(cases)))
    random.Random(seed).shuffle(idx)
    n = len(cases)
    n_val = max(1, round(n * split["val"]))
    n_test = max(1, round(n * split["test"]))
    n_train = n - n_val - n_test
    pick = lambda ids: [cases[i] for i in ids]
    return pick(idx[:n_train]), pick(idx[n_train:n_train + n_val]), pick(idx[n_train + n_val:])


def get_loaders(cfg):
    d, t = cfg["data"], cfg["train"]
    root = find_dataset_root(cfg["paths"]["data_root"])
    cases = list_cases(root)
    if d.get("max_cases"):
        cases = cases[: int(d["max_cases"])]
    train, val, test = make_splits(cases, d["split"], cfg["project"]["seed"])

    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "split.json", "w") as f:
        json.dump({"train": [c["id"] for c in train], "val": [c["id"] for c in val],
                   "test": [c["id"] for c in test]}, f, indent=1)
    print(f"cases: {len(cases)} -> train {len(train)} | val {len(val)} | test {len(test)}  ({root})")

    nw = d.get("num_workers", 0)
    ds_train = CacheDataset(train, get_train_transforms(cfg), cache_rate=d["cache_rate"], num_workers=max(nw, 1))
    vt = get_val_transforms(cfg)
    ds_val = CacheDataset(val, vt, cache_rate=d["cache_rate"], num_workers=max(nw, 1))
    ds_test = CacheDataset(test, vt, cache_rate=d["cache_rate"], num_workers=max(nw, 1))
    train_loader = DataLoader(ds_train, batch_size=t["batch_size"], shuffle=True, num_workers=nw)
    val_loader = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=nw)
    test_loader = DataLoader(ds_test, batch_size=1, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader

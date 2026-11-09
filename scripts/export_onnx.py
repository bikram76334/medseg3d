#!/usr/bin/env python3
"""Export a trained checkpoint to ONNX and verify it against PyTorch.

python scripts/export_onnx.py --ckpt outputs/checkpoints/best.pt --out outputs/onnx/model.onnx
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from models.factory import build_model
from utils.common import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    ap.add_argument("--out", default="outputs/onnx/model.onnx")
    ap.add_argument("--config", default="config/config.yaml")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cpu")
    cfg = ck.get("cfg") or load_config(a.config)       # use the config the model was trained with
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.eval()

    roi = tuple(cfg["infer"]["roi_size"])
    dummy = torch.randn(1, cfg["data"]["in_channels"], *roi)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=17,
              dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, a.out, dynamo=False, **kw)
    except TypeError:                                   # older PyTorch has no dynamo argument
        torch.onnx.export(model, dummy, a.out, **kw)

    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    x = torch.randn(2, cfg["data"]["in_channels"], *roi)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    size = Path(a.out).stat().st_size / 1e6
    print(f"saved {a.out} ({size:.1f} MB) | max abs diff vs PyTorch: {diff:.2e} "
          f"| {'OK' if diff < 1e-3 else 'CHECK'}")


if __name__ == "__main__":
    main()

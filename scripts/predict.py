
import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = dict(
    spacing=(1.5, 1.5, 2.0), roi=(96, 96, 96), overlap=0.5, sw_batch=2,
    mode="ct_window", a_min=-57.0, a_max=164.0,
    num_classes=2, in_channels=1, base=16, levels=5, dropout=0.1,
)

PALETTE = np.array([[0, 0, 0], [255, 0, 0], [0, 200, 0], [0, 100, 255],
                    [255, 200, 0], [200, 0, 255]], dtype=np.uint8)


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------
def load_defaults(cfg_path=None):
    """Read config/config.yaml if present; fall back to built-in defaults."""
    cfg = dict(DEFAULTS)
    p = Path(cfg_path) if cfg_path else ROOT / "config" / "config.yaml"
    try:
        import yaml
        if p.exists():
            c = yaml.safe_load(open(p, "r"))
            d, i, m = c["data"], c["infer"], c.get("model", {})
            inten = d.get("intensity", {})
            cfg.update(
                spacing=tuple(d["spacing"]), num_classes=d["num_classes"],
                in_channels=d.get("in_channels", 1),
                mode=inten.get("mode", cfg["mode"]),
                a_min=float(inten.get("a_min", cfg["a_min"])),
                a_max=float(inten.get("a_max", cfg["a_max"])),
                roi=tuple(i["roi_size"]), overlap=float(i["overlap"]),
                sw_batch=int(i["sw_batch_size"]),
                base=m.get("base_channels", cfg["base"]),
                levels=m.get("levels", cfg["levels"]),
                dropout=m.get("dropout", cfg["dropout"]),
            )
    except Exception as e:                      # config problems should not stop inference
        print(f"[warn] could not read config ({e}); using defaults", file=sys.stderr)
    return cfg


# ----------------------------------------------------------------------------
# model backends
# ----------------------------------------------------------------------------
class Segmenter:
    """Callable: float32 batch [B,1,D,H,W] -> logits [B,C,D,H,W]."""

    def __init__(self, backend, model_path=None, cfg=None):
        self.backend = backend
        self.cfg = cfg or load_defaults()
        if backend == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        elif backend == "torch":
            import torch
            sys.path.insert(0, str(ROOT))
            from models.unet3d import UNet3D
            ckpt = torch.load(str(model_path), map_location="cpu")
            if isinstance(ckpt, torch.nn.Module):
                self.net = ckpt
            else:
                state = ckpt
                if isinstance(ckpt, dict):
                    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
                self.net = UNet3D(self.cfg["in_channels"], self.cfg["num_classes"],
                                  self.cfg["base"], self.cfg["levels"], self.cfg["dropout"])
                self.net.load_state_dict(state)
            self.net.eval()
            self.torch = torch
        elif backend == "demo":
            pass                                  # simple intensity threshold, for testing only
        else:
            raise ValueError(f"unknown backend: {backend}")

    def __call__(self, batch):
        if self.backend == "onnx":
            return self.sess.run(None, {self.inp: batch.astype(np.float32)})[0]
        if self.backend == "torch":
            with self.torch.no_grad():
                return self.net(self.torch.from_numpy(batch)).numpy()
        x = batch[:, 0]                           # demo: class 1 where normalized intensity > 0.5
        return np.stack([(0.5 - x) * 10, (x - 0.5) * 10], axis=1).astype(np.float32)


def make_segmenter(model_path=None, demo=False, cfg=None):
    if demo:
        return Segmenter("demo", cfg=cfg)
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    backend = "onnx" if str(model_path).lower().endswith(".onnx") else "torch"
    return Segmenter(backend, model_path, cfg)


# ----------------------------------------------------------------------------
# pre-processing
# ----------------------------------------------------------------------------
def _fix_shape(a, shape):
    """Crop or zero-pad a to exactly `shape`."""
    out = np.zeros(shape, dtype=a.dtype)
    sl_in = tuple(slice(0, min(s, t)) for s, t in zip(a.shape, shape))
    out[sl_in] = a[sl_in]
    return out


def normalize(x, cfg):
    if cfg["mode"] == "ct_window":
        lo, hi = cfg["a_min"], cfg["a_max"]
        return ((np.clip(x, lo, hi) - lo) / (hi - lo)).astype(np.float32)
    nz = x != 0                                   # z-score over non-zero voxels (MRI)
    if nz.sum() < 2:
        return np.zeros_like(x, dtype=np.float32)
    m, s = x[nz].mean(), x[nz].std() + 1e-8
    out = np.zeros_like(x, dtype=np.float32)
    out[nz] = (x[nz] - m) / s
    return out


def preprocess(path, cfg):
    orig = nib.load(str(path))
    canon = nib.as_closest_canonical(orig)        # RAS orientation
    data = np.asarray(canon.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    spacing = np.array(canon.header.get_zooms()[:3], dtype=float)
    factors = spacing / np.array(cfg["spacing"], dtype=float)
    res = ndi.zoom(data, factors, order=1)
    return dict(orig=orig, canon=canon, canon_data=data, spacing=spacing,
                x=normalize(res, cfg))


# ----------------------------------------------------------------------------
# sliding-window inference
# ----------------------------------------------------------------------------
def gaussian_weight(roi, sigma_scale=0.125):
    ws = []
    for r in roi:
        x = np.arange(r) - (r - 1) / 2.0
        ws.append(np.exp(-(x ** 2) / (2 * (r * sigma_scale) ** 2)))
    w = ws[0][:, None, None] * ws[1][None, :, None] * ws[2][None, None, :]
    return np.clip(w / w.max(), 1e-3, None).astype(np.float32)


def _starts(size, roi, overlap):
    if size <= roi:
        return [0]
    step = max(1, int(roi * (1 - overlap)))
    s = list(range(0, size - roi + 1, step))
    if s[-1] != size - roi:
        s.append(size - roi)
    return s


def _softmax(z, axis=1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sliding_window(vol, seg, roi, overlap=0.5, sw_batch=2, progress=None):
    """vol: [D,H,W] float32 -> label map [D,H,W] uint8."""
    orig_shape = vol.shape
    pad = [max(0, r - s) for s, r in zip(vol.shape, roi)]
    v = np.pad(vol, [(0, p) for p in pad]) if any(pad) else vol
    coords = [(a, b, c) for a in _starts(v.shape[0], roi[0], overlap)
              for b in _starts(v.shape[1], roi[1], overlap)
              for c in _starts(v.shape[2], roi[2], overlap)]
    w = gaussian_weight(roi)
    prob, norm = None, np.zeros(v.shape, dtype=np.float32)
    rd, rh, rw = roi
    for i in range(0, len(coords), sw_batch):
        chunk = coords[i:i + sw_batch]
        batch = np.stack([v[a:a + rd, b:b + rh, c:c + rw] for a, b, c in chunk])[:, None]
        p = _softmax(seg(batch))
        if prob is None:
            prob = np.zeros((p.shape[1],) + v.shape, dtype=np.float32)
        for (a, b, c), pi in zip(chunk, p):
            prob[:, a:a + rd, b:b + rh, c:c + rw] += pi * w
            norm[a:a + rd, b:b + rh, c:c + rw] += w
        if progress:
            progress(min(1.0, (i + len(chunk)) / len(coords)))
    prob /= norm
    labels = prob.argmax(axis=0).astype(np.uint8)
    return labels[:orig_shape[0], :orig_shape[1], :orig_shape[2]]


# ----------------------------------------------------------------------------
# post-processing and saving
# ----------------------------------------------------------------------------
def postprocess(labels, keep_largest=True, fill_holes=True):
    out = np.zeros_like(labels)
    for c in range(1, int(labels.max()) + 1):
        m = labels == c
        if not m.any():
            continue
        if keep_largest:
            cc, n = ndi.label(m)
            if n > 1:
                sizes = ndi.sum(m, cc, range(1, n + 1))
                m = cc == (1 + int(np.argmax(sizes)))
        if fill_holes:
            m = ndi.binary_fill_holes(m)
        out[m & (out == 0)] = c
    return out


def segment_file(path, seg, cfg=None, postproc=True, progress=None):
    """Full pipeline. Returns image + mask in the ORIGINAL orientation of the input file."""
    cfg = cfg or load_defaults()
    pre = preprocess(path, cfg)
    labels = sliding_window(pre["x"], seg, cfg["roi"], cfg["overlap"], cfg["sw_batch"], progress)
    if postproc:
        labels = postprocess(labels)
    # back to the canonical grid (nearest neighbour), then to the original orientation
    cshape = pre["canon_data"].shape
    back = ndi.zoom(labels, np.array(cshape) / np.array(labels.shape), order=0)
    mask_canon = _fix_shape(back, cshape).astype(np.uint8)
    trans = ornt_transform(io_orientation(pre["canon"].affine), io_orientation(pre["orig"].affine))
    mask = apply_orientation(mask_canon, trans).astype(np.uint8)
    image = apply_orientation(pre["canon_data"], trans)
    spacing = pre["orig"].header.get_zooms()[:3]
    return dict(image=image, mask=mask, spacing=spacing, affine=pre["orig"].affine,
                header=pre["orig"].header)


def save_mask(mask, affine, path):
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def class_volumes(mask, spacing):
    """Return {class: (voxels, millilitres)}."""
    vox = float(np.prod(spacing))
    return {int(c): (int((mask == c).sum()), float((mask == c).sum()) * vox / 1000.0)
            for c in range(1, int(mask.max()) + 1)}


# ----------------------------------------------------------------------------
# overlay rendering (numpy only)
# ----------------------------------------------------------------------------
AXES = {"sagittal": 0, "coronal": 1, "axial": 2}


def best_slice(mask, plane="axial"):
    axis = AXES[plane]
    other = tuple(a for a in range(3) if a != axis)
    counts = (mask > 0).sum(axis=other)
    return int(np.argmax(counts)) if counts.max() > 0 else mask.shape[axis] // 2


def render_overlay(image, mask, plane="axial", idx=None, alpha=0.4):
    axis = AXES[plane]
    n = image.shape[axis]
    idx = n // 2 if idx is None else int(np.clip(idx, 0, n - 1))
    sl, ms = np.take(image, idx, axis=axis), np.take(mask, idx, axis=axis)
    sub = image[::4, ::4, ::4]
    lo, hi = np.percentile(sub, (1, 99))
    g = np.clip((sl - lo) / (hi - lo + 1e-8), 0, 1)
    rgb = np.stack([g, g, g], axis=-1) * 255.0
    col = PALETTE[np.clip(ms, 0, len(PALETTE) - 1)].astype(np.float32)
    m = (ms > 0)[..., None]
    rgb = np.where(m, (1 - alpha) * rgb + alpha * col, rgb)
    return np.rot90(rgb).astype(np.uint8)


# ----------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="3D segmentation inference (ONNX / PyTorch)")
    ap.add_argument("--input", required=True, help="NIfTI scan (.nii / .nii.gz)")
    ap.add_argument("--output", required=True, help="where to save the mask (.nii.gz)")
    ap.add_argument("--model", default=str(ROOT / "outputs" / "onnx" / "model.onnx"),
                    help=".onnx or .pt file")
    ap.add_argument("--config", default=None)
    ap.add_argument("--demo", action="store_true", help="no model: threshold demo for testing")
    ap.add_argument("--overlay", default=None, help="optional PNG with the mask on the best slice")
    ap.add_argument("--plane", default="axial", choices=list(AXES))
    ap.add_argument("--no-postprocess", action="store_true")
    a = ap.parse_args()

    cfg = load_defaults(a.config)
    seg = make_segmenter(a.model, a.demo, cfg)
    res = segment_file(a.input, seg, cfg, postproc=not a.no_postprocess,
                       progress=lambda f: print(f"\rinference {f * 100:5.1f}%", end="", flush=True))
    print()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    save_mask(res["mask"], res["affine"], a.output)
    print(f"saved mask: {a.output}  shape={res['mask'].shape}")
    for c, (n, ml) in class_volumes(res["mask"], res["spacing"]).items():
        print(f"  class {c}: {n} voxels, {ml:.2f} mL")
    if a.overlay:
        import matplotlib.image as mpimg
        k = best_slice(res["mask"], a.plane)
        mpimg.imsave(a.overlay, render_overlay(res["image"], res["mask"], a.plane, k))
        print(f"saved overlay: {a.overlay} ({a.plane} slice {k})")


if __name__ == "__main__":
    main()
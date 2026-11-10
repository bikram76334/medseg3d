#!/usr/bin/env python3
"""
app/gradio_app.py - small web interface for the segmentation pipeline.

Run from the project root:
    python app/gradio_app.py
then open http://127.0.0.1:7860

Upload a NIfTI scan, pick a model (.onnx or .pt), click Run, scroll through the slices
with the overlay, and download the mask. Uses scripts/predict.py as the engine.
"""
import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import predict as P  # noqa: E402

CFG = P.load_defaults()
DEFAULT_MODEL = ROOT / "outputs" / "onnx" / "model.onnx"


def _path(f):
    """gradio returns a str path (4+/5) or a tempfile object (3.x)."""
    if f is None:
        return None
    return f if isinstance(f, str) else getattr(f, "name", None)


def run(scan, model_file, use_demo, mode, a_min, a_max, overlap, plane, alpha,
        progress=gr.Progress()):
    scan_path = _path(scan)
    if scan_path is None:
        raise gr.Error("Upload a NIfTI file (.nii or .nii.gz) first.")

    cfg = dict(CFG)
    cfg.update(mode="ct_window" if mode.startswith("CT") else "zscore",
               a_min=float(a_min), a_max=float(a_max), overlap=float(overlap))

    model_path = _path(model_file) or str(DEFAULT_MODEL)
    try:
        seg = P.make_segmenter(model_path, demo=use_demo, cfg=cfg)
    except FileNotFoundError:
        raise gr.Error(f"No model found at {model_path}. Upload a .onnx / .pt file, "
                       f"or tick 'Demo mode' to test the interface.")
    except Exception as e:
        raise gr.Error(f"Could not load the model: {e}")

    try:
        res = P.segment_file(scan_path, seg, cfg,
                             progress=lambda f: progress(f, desc="Sliding-window inference"))
    except Exception as e:
        raise gr.Error(f"Segmentation failed: {e}")

    out_dir = Path(tempfile.mkdtemp())
    out_file = out_dir / "mask.nii.gz"
    P.save_mask(res["mask"], res["affine"], out_file)

    state = dict(image=res["image"], mask=res["mask"])
    k = P.best_slice(res["mask"], plane)
    n = res["image"].shape[P.AXES[plane]]
    img = P.render_overlay(res["image"], res["mask"], plane, k, alpha)

    lines = [f"Shape: {res['mask'].shape}   Voxel size (mm): "
             f"{tuple(round(float(s), 2) for s in res['spacing'])}"]
    vols = P.class_volumes(res["mask"], res["spacing"])
    if vols:
        lines += [f"Class {c}: {nv} voxels = {ml:.2f} mL" for c, (nv, ml) in vols.items()]
    else:
        lines.append("No foreground predicted.")
    if use_demo:
        lines.append("DEMO MODE: simple intensity threshold, NOT a trained model.")

    png = out_dir / "overlay_montage.png"
    P.save_montage(res["image"], res["mask"], png, plane, 8, alpha)

    return (img, gr.update(minimum=0, maximum=n - 1, value=k, visible=True),
            "\n".join(lines), str(out_file), str(png), state)


def show(state, plane, idx, alpha):
    if not state:
        return None
    return P.render_overlay(state["image"], state["mask"], plane, idx, alpha)


def change_plane(state, plane, alpha):
    if not state:
        return None, gr.update()
    k = P.best_slice(state["mask"], plane)
    n = state["image"].shape[P.AXES[plane]]
    return (P.render_overlay(state["image"], state["mask"], plane, k, alpha),
            gr.update(minimum=0, maximum=n - 1, value=k))


def build_ui():
    with gr.Blocks(title="3D Medical Image Segmentation") as demo:
        gr.Markdown("# 3D Medical Image Segmentation\n"
                    "Upload a NIfTI scan, run the 3D U-Net, and inspect the mask. "
                    "Research prototype - not for clinical use.")
        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                scan = gr.File(label="Scan (.nii / .nii.gz)", file_types=[".nii", ".gz"])
                model = gr.File(label="Model (.onnx or .pt). Leave empty to use outputs/onnx/model.onnx",
                                file_types=[".onnx", ".pt", ".pth"])
                demo_mode = gr.Checkbox(label="Demo mode (no model, threshold only, for testing the UI)",
                                        value=False)
                mode = gr.Radio(["CT (window)", "MRI (z-score)"],
                                value="CT (window)" if CFG["mode"] == "ct_window" else "MRI (z-score)",
                                label="Intensity normalization")
                with gr.Row():
                    a_min = gr.Number(value=CFG["a_min"], label="CT window min (HU)")
                    a_max = gr.Number(value=CFG["a_max"], label="CT window max (HU)")
                overlap = gr.Slider(0.0, 0.9, value=CFG["overlap"], step=0.05,
                                    label="Sliding-window overlap")
                plane = gr.Radio(["axial", "coronal", "sagittal"], value="axial", label="View")
                alpha = gr.Slider(0.1, 0.9, value=0.4, step=0.05, label="Overlay opacity")
                run_btn = gr.Button("Run segmentation", variant="primary")
            with gr.Column(scale=2):
                view = gr.Image(label="Overlay", type="numpy", interactive=False)
                slice_idx = gr.Slider(0, 1, value=0, step=1, label="Slice", visible=False)
                info = gr.Textbox(label="Result", lines=5, interactive=False)
                mask_file = gr.File(label="Download mask (.nii.gz)", interactive=False)
                montage_file = gr.File(label="Download overlay image (large PNG, 8 slices)", interactive=False)

        run_btn.click(run,
                      [scan, model, demo_mode, mode, a_min, a_max, overlap, plane, alpha],
                      [view, slice_idx, info, mask_file, montage_file, state])
        slice_idx.change(show, [state, plane, slice_idx, alpha], view)
        alpha.change(show, [state, plane, slice_idx, alpha], view)
        plane.change(change_plane, [state, plane, alpha], [view, slice_idx])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    a = ap.parse_args()
    build_ui().queue().launch(server_port=a.port, share=a.share)
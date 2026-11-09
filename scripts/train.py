#!/usr/bin/env python3
"""
scripts/train.py - train the 3D U-Net, validate with Dice, keep the best model, test at the end.

Examples
--------
python scripts/train.py --config config/config.yaml
python scripts/train.py --set train.epochs=2 --set data.max_cases=20          # quick debug run
python scripts/train.py --eval-only --ckpt outputs/checkpoints/best.pt        # only test-set evaluation
Kaggle: python scripts/train.py --data-root /kaggle/input/<ds>/Task04_Hippocampus \
        --out-dir /kaggle/working/outputs --set data.num_workers=2
"""
import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference

from data.datamodule import get_loaders
from evaluation.metrics import dice_score, evaluate_case
from evaluation.report import flatten, print_summary, summarize, write_report
from losses.dice_ce import DiceCELoss
from models.factory import build_model
from utils.common import apply_overrides, ensure_dirs, get_device, load_config, set_out_dir, set_seed
from visualization.overlay import export_overlays


@torch.no_grad()
def predict_volume(model, image, cfg, device, use_amp):
    d = cfg["infer"]
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = sliding_window_inference(image.to(device), tuple(d["roi_size"]), d["sw_batch_size"],
                                          model, overlap=d["overlap"], mode=d["blend"])
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def validate(model, loader, cfg, device, use_amp):
    """Mean Dice per foreground class over the whole validation set."""
    model.eval()
    nc = cfg["data"]["num_classes"]
    scores = []
    for batch in loader:
        pred = predict_volume(model, batch["image"], cfg, device, use_amp)
        gt = batch["label"][0, 0].numpy().astype(np.uint8)
        scores.append([dice_score(pred == c, gt == c) for c in range(1, nc)])
    return np.mean(scores, axis=0)                # shape [num_classes-1]


def test_evaluation(model, loader, cfg, device, use_amp):
    """Full metrics (Dice, IoU, HD, HD95) on the test split + a few overlay examples."""
    model.eval()
    nc, sp = cfg["data"]["num_classes"], cfg["data"]["spacing"]
    rows, store = [], []
    for batch in loader:
        pred = predict_volume(model, batch["image"], cfg, device, use_amp)
        gt = batch["label"][0, 0].numpy().astype(np.uint8)
        cid = batch["id"][0] if "id" in batch else f"case{len(rows)}"
        rows.append(flatten(cid, evaluate_case(pred, gt, sp, nc)))
        store.append((batch["image"][0, 0].numpy(), pred))
    log_dir = Path(cfg["paths"]["log_dir"])
    write_report(rows, log_dir / "test_metrics.csv")
    summ = summarize(rows)
    print("\nTEST SET RESULTS")
    print_summary(summ)
    order = np.argsort([r["mean_dice"] for r in rows])
    for tag, i in (("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])):
        out = Path(cfg["paths"]["overlay_dir"]) / f"{tag}_{rows[i]['id']}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        export_overlays(store[i][0], store[i][1], str(out), n_slices=cfg["viz"]["n_slices"],
                        alpha=cfg["viz"]["alpha"])
    print(f"saved: {log_dir / 'test_metrics.csv'} and overlays in {cfg['paths']['overlay_dir']}")
    return summ


def plot_curves(csv_path, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ep, loss, vep, vdice = [], [], [], []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"])); loss.append(float(r["train_loss"]))
            if r["mean_dice"]:
                vep.append(int(r["epoch"])); vdice.append(float(r["mean_dice"]))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ep, loss); ax[0].set_title("Training loss"); ax[0].set_xlabel("epoch")
    ax[1].plot(vep, vdice); ax[1].set_title("Validation mean Dice"); ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. train.epochs=2")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default=None, help="re-root all outputs (e.g. /kaggle/working/outputs)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--ckpt", default=None, help="checkpoint for --eval-only (default best.pt)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.data_root:
        cfg["paths"]["data_root"] = a.data_root
    if a.out_dir:
        set_out_dir(cfg, a.out_dir)
    apply_overrides(cfg, a.set)
    set_seed(cfg["project"]["seed"])
    ensure_dirs(cfg)
    device = get_device()
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    print(f"device: {device} | AMP: {use_amp}")

    ck_dir, log_dir = Path(cfg["paths"]["ckpt_dir"]), Path(cfg["paths"]["log_dir"])
    with open(log_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    train_loader, val_loader, test_loader = get_loaders(cfg)
    model = build_model(cfg).to(device)
    print(f"model: {cfg['model']['name']} | parameters: {sum(p.numel() for p in model.parameters()):,}")

    if a.eval_only:
        ck = torch.load(a.ckpt or ck_dir / "best.pt", map_location=device)
        model.load_state_dict(ck["model"])
        test_evaluation(model, test_loader, cfg, device, use_amp)
        return

    t = cfg["train"]
    crit = DiceCELoss(t["loss"]["dice_weight"], t["loss"]["ce_weight"])
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start, best = 1, -1.0
    last = ck_dir / "last.pt"
    if t.get("resume") and last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best dice {best:.4f})")

    log_csv = log_dir / "train_log.csv"
    nc = cfg["data"]["num_classes"]
    cols = ["epoch", "train_loss", "lr", "mean_dice"] + [f"dice_c{c}" for c in range(1, nc)] + ["seconds"]
    if not (t.get("resume") and log_csv.exists()):
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    for epoch in range(start, t["epochs"] + 1):
        t0 = time.time()
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device).squeeze(1).long()
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x).float(), y)
            scaler.scale(loss).backward()
            if t.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        mean_d, per = "", [""] * (nc - 1)
        if epoch % t["val_interval"] == 0 or epoch == t["epochs"]:
            per_cls = validate(model, val_loader, cfg, device, use_amp)
            mean_d, per = float(per_cls.mean()), [float(v) for v in per_cls]
            if mean_d > best:
                best = mean_d
                torch.save({"model": model.state_dict(), "epoch": epoch, "best_dice": best, "cfg": cfg},
                           ck_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "cfg": cfg}, last)

        row = [epoch, float(np.mean(losses)), opt.param_groups[0]["lr"], mean_d] + per + [round(time.time() - t0, 1)]
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow(row)
        msg = f"epoch {epoch:3d}/{t['epochs']} loss {np.mean(losses):.4f}"
        if mean_d != "":
            msg += f" | val dice {mean_d:.4f} (best {best:.4f})"
        print(msg + f" | {time.time() - t0:.0f}s", flush=True)

    plot_curves(log_csv, log_dir / "curves.png")
    print(f"\nbest validation Dice: {best:.4f}  -> {ck_dir / 'best.pt'}")
    ck = torch.load(ck_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    test_evaluation(model, test_loader, cfg, device, use_amp)


if __name__ == "__main__":
    main()

"""Per-case CSV + mean/std summary."""
import csv
from pathlib import Path

import numpy as np


def flatten(case_id, metrics):
    """metrics: {class: {dice, iou, hd, hd95}} -> one flat row."""
    row = {"id": case_id}
    for c, m in metrics.items():
        for k, v in m.items():
            row[f"c{c}_{k}"] = v
    row["mean_dice"] = float(np.mean([m["dice"] for m in metrics.values()]))
    return row


def write_report(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    """mean +/- std per metric column; infinite values (empty masks) are counted separately."""
    out = {}
    for k in rows[0]:
        if k == "id":
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        ok = np.isfinite(v)
        out[k] = (float(v[ok].mean()) if ok.any() else float("nan"),
                  float(v[ok].std()) if ok.any() else float("nan"), int((~ok).sum()))
    return out


def print_summary(summary):
    print(f"{'metric':<14}{'mean':>10}{'std':>10}{'n_inf':>7}")
    for k, (m, s, n) in summary.items():
        print(f"{k:<14}{m:>10.4f}{s:>10.4f}{n:>7}")

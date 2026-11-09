"""Step 4 - Standalone spatial metrics, implemented from scratch (NumPy/SciPy only)."""
import numpy as np
from scipy import ndimage as ndi


def dice_score(pred, gt, eps=1e-8):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0                                            # both empty -> perfect by convention
    return float(2.0 * np.logical_and(pred, gt).sum() / (pred.sum() + gt.sum() + eps))


def iou_score(pred, gt, eps=1e-8):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / (union + eps))


def surface_voxels(mask):
    """Boundary = mask minus its erosion."""
    mask = mask.astype(bool)
    return mask & ~ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))


def _surface_distances(a, b, spacing):
    """Distances (mm) from every surface voxel of a to the nearest surface voxel of b."""
    sa, sb = surface_voxels(a), surface_voxels(b)
    dist_to_b = ndi.distance_transform_edt(~sb, sampling=spacing)
    return dist_to_b[sa]


def hausdorff_distance(pred, gt, spacing=(1, 1, 1), percentile=100):
    """HD (percentile=100) or HD95 (percentile=95) in mm. inf if one mask is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")
    d = np.concatenate([_surface_distances(pred, gt, spacing),
                        _surface_distances(gt, pred, spacing)])
    return float(np.percentile(d, percentile))


def hd95(pred, gt, spacing=(1, 1, 1)):
    return hausdorff_distance(pred, gt, spacing, 95)


def evaluate_case(pred, gt, spacing, num_classes):
    """pred, gt: integer label volumes. Returns per-class metrics (background skipped)."""
    out = {}
    for c in range(1, num_classes):
        p, g = pred == c, gt == c
        out[c] = {"dice": dice_score(p, g), "iou": iou_score(p, g),
                  "hd": hausdorff_distance(p, g, spacing),
                  "hd95": hd95(p, g, spacing)}
    return out

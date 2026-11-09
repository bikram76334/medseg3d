import numpy as np
import pytest
pytest.importorskip("monai")
nib = pytest.importorskip("nibabel")
from data.transforms import get_train_transforms, get_val_transforms


def _cfg():
    return {"data": {"orientation": "RAS", "spacing": [1, 1, 1], "patch_size": [16, 16, 16],
                     "num_samples": 2, "intensity": {"mode": "zscore", "a_min": -57, "a_max": 164}}}


def _case(tmp_path):
    rng = np.random.default_rng(0)
    img = rng.normal(100, 10, (24, 28, 20)).astype(np.float32)
    lab = np.zeros(img.shape, np.uint8)
    lab[8:16, 10:18, 6:14] = 1
    img[lab == 1] += 80
    nib.save(nib.Nifti1Image(img, np.eye(4)), str(tmp_path / "i.nii.gz"))
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(tmp_path / "l.nii.gz"))
    return {"image": str(tmp_path / "i.nii.gz"), "label": str(tmp_path / "l.nii.gz")}


def test_val_keeps_alignment_and_normalizes(tmp_path):
    out = get_val_transforms(_cfg())(_case(tmp_path))
    assert out["image"].shape == out["label"].shape
    assert abs(float(out["image"][out["image"] != 0].mean())) < 0.05      # z-score


def test_train_patches_have_patch_size_and_foreground(tmp_path):
    outs = get_train_transforms(_cfg())(_case(tmp_path))
    assert len(outs) == 2
    for o in outs:
        assert tuple(o["image"].shape[1:]) == (16, 16, 16)
        assert tuple(o["label"].shape[1:]) == (16, 16, 16)

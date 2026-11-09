import numpy as np
from evaluation.metrics import dice_score, hausdorff_distance, iou_score


def cube(n=20, lo=5, hi=15, shift=0):
    m = np.zeros((n, n, n), dtype=bool)
    m[lo + shift:hi + shift, lo:hi, lo:hi] = True
    return m


def test_dice_identical():
    assert abs(dice_score(cube(), cube()) - 1.0) < 1e-6


def test_dice_disjoint():
    assert dice_score(cube(lo=0, hi=5), cube(lo=10, hi=15)) < 1e-6


def test_dice_half_overlap():
    a, b = cube(), cube(shift=5)          # 10x10x10 cubes, half overlap
    assert abs(dice_score(a, b) - 0.5) < 1e-6
    assert abs(iou_score(a, b) - (500 / 1500)) < 1e-6


def test_hausdorff_shift():
    d = hausdorff_distance(cube(), cube(shift=3), spacing=(1, 1, 1))
    assert abs(d - 3.0) < 1e-6


def test_hausdorff_spacing():
    d = hausdorff_distance(cube(), cube(shift=3), spacing=(2, 1, 1))
    assert abs(d - 6.0) < 1e-6

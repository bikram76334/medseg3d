"""Step 2 - MONAI transform chains (train / val / infer)."""
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    NormalizeIntensityd, ScaleIntensityRanged, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandAffined, RandScaleIntensityd, RandShiftIntensityd,
    RandGaussianNoised, EnsureTyped,
)

KEYS = ["image", "label"]


def _base(cfg):
    d = cfg["data"]
    t = [
        LoadImaged(keys=KEYS),
        EnsureChannelFirstd(keys=KEYS),
        Orientationd(keys=KEYS, axcodes=d.get("orientation", "RAS")),
        Spacingd(keys=KEYS, pixdim=tuple(d["spacing"]), mode=("bilinear", "nearest")),
    ]
    inten = d["intensity"]
    if inten["mode"] == "ct_window":
        t.append(ScaleIntensityRanged(keys=["image"], a_min=inten["a_min"], a_max=inten["a_max"],
                                      b_min=0.0, b_max=1.0, clip=True))
    else:                                         # MRI: z-score over non-zero voxels
        t.append(NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True))
    return t


def get_train_transforms(cfg):
    d = cfg["data"]
    patch = tuple(d["patch_size"])
    return Compose(_base(cfg) + [
        SpatialPadd(keys=KEYS, spatial_size=patch),
        RandCropByPosNegLabeld(keys=KEYS, label_key="label", spatial_size=patch,
                               pos=1, neg=1, num_samples=d["num_samples"]),
        # flip only left-right (axis 0): flipping anterior-posterior would swap the two labels
        RandFlipd(keys=KEYS, prob=0.5, spatial_axis=0),
        RandAffined(keys=KEYS, prob=0.3, rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest"), padding_mode="border"),
        RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        EnsureTyped(keys=KEYS),
    ])


def get_val_transforms(cfg):
    """Whole volumes, no augmentation (used for validation and test)."""
    return Compose(_base(cfg) + [EnsureTyped(keys=KEYS)])


def get_infer_transforms(cfg):
    return get_val_transforms(cfg)

"""Build a model from config."""
from models.unet3d import UNet3D


def build_model(cfg):
    m, d = cfg["model"], cfg["data"]
    if m["name"] == "unet3d":
        return UNet3D(d["in_channels"], d["num_classes"], m["base_channels"],
                      m["levels"], m.get("dropout", 0.0))
    if m["name"] == "monai_unet":                 # baseline for comparison
        from monai.networks.nets import UNet
        b, n = m["base_channels"], m["levels"]
        return UNet(spatial_dims=3, in_channels=d["in_channels"], out_channels=d["num_classes"],
                    channels=tuple(b * 2 ** i for i in range(n)), strides=(2,) * (n - 1),
                    num_res_units=2)
    raise ValueError(f"unknown model {m['name']}")

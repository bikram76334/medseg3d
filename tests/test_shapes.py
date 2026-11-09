import pytest
torch = pytest.importorskip("torch")
from models.unet3d import UNet3D
from losses.dice_ce import DiceCELoss


def test_unet_output_shape():
    m = UNet3D(in_channels=1, num_classes=3, base=8, levels=4)
    y = m(torch.randn(1, 1, 32, 32, 32))
    assert y.shape == (1, 3, 32, 32, 32)


def test_loss_is_finite_and_decreases_for_perfect_prediction():
    target = torch.randint(0, 3, (2, 16, 16, 16))
    perfect = torch.nn.functional.one_hot(target, 3).permute(0, 4, 1, 2, 3).float() * 20
    random = torch.randn(2, 3, 16, 16, 16)
    crit = DiceCELoss()
    assert torch.isfinite(crit(random, target))
    assert crit(perfect, target) < crit(random, target)

"""Step 3 - 3D U-Net from scratch (encoder - bottleneck - decoder + skips)."""
import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.PReLU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True), nn.PReLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, base=16, levels=5, dropout=0.1):
        super().__init__()
        ch = [base * 2 ** i for i in range(levels)]          # e.g. 16,32,64,128,256
        self.enc = nn.ModuleList()
        prev = in_channels
        for c in ch:
            self.enc.append(ConvBlock3D(prev, c, dropout))
            prev = c
        self.pool = nn.MaxPool3d(2)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(levels - 1, 0, -1):
            self.up.append(nn.ConvTranspose3d(ch[i], ch[i - 1], 2, stride=2))
            self.dec.append(ConvBlock3D(ch[i - 1] * 2, ch[i - 1], dropout))
        self.head = nn.Conv3d(ch[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:                         # last = bottleneck
                skips.append(x)
                x = self.pool(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = dec(torch.cat([skip, x], dim=1))              # skip connection
        return self.head(x)                                   # raw logits


if __name__ == "__main__":
    m = UNet3D()
    y = m(torch.randn(1, 1, 64, 64, 64))
    print("output:", tuple(y.shape), "| params:", sum(p.numel() for p in m.parameters()))

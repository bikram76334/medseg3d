"""Step 6 - transparent mask overlays on anatomical slices (PNG grid / GIF)."""
import numpy as np
import matplotlib.pyplot as plt


def overlay_slice(ax, image_slice, mask_slice, alpha=0.4):
    ax.imshow(np.rot90(image_slice), cmap="gray")
    masked = np.ma.masked_where(np.rot90(mask_slice) == 0, np.rot90(mask_slice))
    ax.imshow(masked, cmap="autumn", vmin=1, vmax=2, alpha=alpha, interpolation="none")
    ax.axis("off")


def export_overlays(image, mask, out_png, n_slices=8, alpha=0.4):
    """Pick the n slices with most mask voxels and save a grid."""
    counts = (mask > 0).sum(axis=(0, 1))
    idx = np.argsort(counts)[-n_slices:][::-1]
    fig, axes = plt.subplots(2, (n_slices + 1) // 2, figsize=(3 * ((n_slices + 1) // 2), 6))
    for ax, k in zip(axes.ravel(), idx):
        overlay_slice(ax, image[:, :, k], mask[:, :, k], alpha)
        ax.set_title(f"slice {k}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

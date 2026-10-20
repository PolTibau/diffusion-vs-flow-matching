"""Plotting helpers: 2D scatter and trajectories, image grids and GIFs."""

import matplotlib.pyplot as plt
import numpy as np
import torch


def scatter(ax, x, title=None, color="C0", s=2, lim=2.6):
    x = torch.as_tensor(x).cpu()
    ax.scatter(x[:, 0], x[:, 1], s=s, c=color, alpha=0.6, linewidths=0)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
    if title:
        ax.set_title(title)


def trajectories(ax, traj, n=300, title=None, lim=2.6):
    """Draws the paths of ``n`` particles from noise (grey) to samples (colour)."""
    path = torch.stack(traj)[:, :n].cpu()  # (time, n, 2)
    for i in range(path.shape[1]):
        ax.plot(path[:, i, 0], path[:, i, 1], color="k", alpha=0.15, lw=0.6)
    ax.scatter(path[0, :, 0], path[0, :, 1], s=3, c="grey", alpha=0.5)
    ax.scatter(path[-1, :, 0], path[-1, :, 1], s=3, c="C3")
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
    if title:
        ax.set_title(title)


def to_image(x: torch.Tensor, nrow: int | None = None) -> np.ndarray:
    """(B, C, H, W) in [-1, 1] -> grid as an (H', W', C) array in [0, 1]."""
    from torchvision.utils import make_grid

    nrow = nrow or int(np.ceil(np.sqrt(len(x))))
    grid = make_grid(x.detach().cpu().clamp(-1, 1), nrow=nrow, padding=2, pad_value=1)
    return ((grid + 1) / 2).permute(1, 2, 0).numpy()


def show_grid(x, title=None, ax=None):
    ax = ax or plt.gca()
    ax.imshow(to_image(x))
    ax.axis("off")
    if title:
        ax.set_title(title)


def save_gif(traj, path, fps=12, hold=10):
    """Saves a denoising trajectory of image batches as a GIF (pauses on the last frame)."""
    import imageio.v2 as imageio

    frames = [(to_image(x) * 255).astype(np.uint8) for x in traj]
    frames += [frames[-1]] * hold
    imageio.mimsave(path, frames, duration=1 / fps, loop=0)

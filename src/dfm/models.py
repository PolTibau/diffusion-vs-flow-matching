"""Networks used by both methods.

Every model has the same interface, ``model(x, t)``, where ``t`` is a float tensor of
shape ``(B,)`` in ``[0, 1]``. Diffusion reads the output as the predicted noise, flow
matching as the predicted velocity: the architecture is shared, only the target changes.
"""

import math

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Maps a scalar time to a vector of sines and cosines (as in Transformers)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
        args = 1000 * t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class MLP(nn.Module):
    """Small time-conditioned MLP for 2D toy data."""

    def __init__(self, dim: int = 2, hidden: int = 256, time_dim: int = 64, depth: int = 4):
        super().__init__()
        self.time_embed = nn.Sequential(SinusoidalEmbedding(time_dim), nn.Linear(time_dim, hidden), nn.SiLU())
        self.inp = nn.Linear(dim, hidden)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.SiLU(), nn.Linear(hidden, hidden)) for _ in range(depth)
        )
        self.out = nn.Sequential(nn.SiLU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.time_embed(t)
        for block in self.blocks:
            h = h + block(h)  # residual connections keep training stable
        return self.out(h)


class UNet(nn.Module):
    """Wraps 🤗 diffusers' ``UNet2DModel`` behind the ``model(x, t)`` interface."""

    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # diffusers expects timesteps in the DDPM range [0, 999]
        return self.unet(x, t * 999).sample


def make_unet(image_size: int = 64, channels: int = 3, base: int = 64) -> UNet:
    """A small U-Net (~15M parameters at base=64) suited to 32x32 or 64x64 images."""
    from diffusers import UNet2DModel

    unet = UNet2DModel(
        sample_size=image_size,
        in_channels=channels,
        out_channels=channels,
        layers_per_block=2,
        block_out_channels=(base, base * 2, base * 2, base * 4),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    return UNet(unet)

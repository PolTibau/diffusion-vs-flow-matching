"""Datasets: 2D toy distributions and the Smithsonian butterflies (🤗 Hub)."""

import math

import numpy as np
import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------- 2D toy data

def sample_2d(name: str, n: int) -> torch.Tensor:
    """Samples ``n`` points from a 2D toy distribution, roughly normalised to [-2, 2]."""
    rng = np.random.default_rng()
    if name == "moons":
        from sklearn.datasets import make_moons

        x, _ = make_moons(n, noise=0.05)
        x = (x - [0.5, 0.25]) * 1.6
    elif name == "spiral":
        theta = np.sqrt(rng.uniform(0, 1, n)) * 3 * math.pi
        r = theta / (3 * math.pi) * 2
        x = np.stack([r * np.cos(theta), r * np.sin(theta)], 1) + rng.normal(0, 0.04, (n, 2))
    elif name == "checkerboard":
        x1 = rng.uniform(-2, 2, n)
        x2 = rng.uniform(0, 1, n) - rng.integers(0, 2, n) * 2 + (np.floor(x1) % 2)
        x = np.stack([x1, x2], 1)
    elif name == "gaussians":
        centers = np.array([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]) * 1.6
        x = centers[rng.integers(0, 8, n)] + rng.normal(0, 0.1, (n, 2))
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return torch.tensor(x, dtype=torch.float32)


def loader_2d(name: str, batch_size: int = 512, n: int = 50_000) -> DataLoader:
    data = sample_2d(name, n)
    return DataLoader(torch.utils.data.TensorDataset(data), batch_size=batch_size, shuffle=True, drop_last=True)


# ---------------------------------------------------------------- images

def loader_butterflies(image_size: int = 64, batch_size: int = 64, num_workers: int = 0) -> DataLoader:
    """~1000 colour butterfly images from the Smithsonian collection, scaled to [-1, 1]."""
    from datasets import load_dataset
    from torchvision import transforms

    ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds.set_transform(lambda batch: {"images": [tf(img.convert("RGB")) for img in batch["image"]]})
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                      num_workers=num_workers, collate_fn=lambda b: (torch.stack([x["images"] for x in b]),))

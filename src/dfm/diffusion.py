"""Denoising diffusion (DDPM, Ho et al. 2020) with DDPM and DDIM samplers.

Convention: t = 0 is data and t = 1 is pure noise. The forward process

    x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,    eps ~ N(0, I)

is the discretisation of a variance-preserving SDE. The network predicts eps, which is
(up to a factor -1/sqrt(1 - abar_t)) the score grad log p_t(x).
"""

import math

import torch
import torch.nn.functional as F


class Diffusion:
    def __init__(self, num_steps: int = 1000, schedule: str = "cosine"):
        self.T = num_steps
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_steps)
        elif schedule == "cosine":  # Nichol & Dhariwal (2021): less noise wasted at the end
            s = 0.008
            f = lambda k: math.cos((k / num_steps + s) / (1 + s) * math.pi / 2) ** 2
            betas = torch.tensor([min(1 - f(k + 1) / f(k), 0.999) for k in range(num_steps)])
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
        self.betas = betas
        self.alphas = 1 - betas
        self.abar = torch.cumprod(self.alphas, 0)

    def _t(self, k: torch.Tensor) -> torch.Tensor:
        """Discrete step k in {0..T-1} -> continuous time in [0, 1] fed to the network."""
        return k.float() / (self.T - 1)

    @staticmethod
    def _bc(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcasts a per-sample scalar to the shape of x."""
        return v.view(-1, *([1] * (x.dim() - 1)))

    def add_noise(self, x0: torch.Tensor, k: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        abar = self._bc(self.abar.to(x0.device)[k], x0)
        return abar.sqrt() * x0 + (1 - abar).sqrt() * eps

    def loss(self, model, x0: torch.Tensor) -> torch.Tensor:
        k = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        eps = torch.randn_like(x0)
        return F.mse_loss(model(self.add_noise(x0, k, eps), self._t(k)), eps)

    @torch.no_grad()
    def sample_ddpm(self, model, shape, device="cpu", keep=None, clip=None):
        """Ancestral sampling: discretised reverse SDE, T stochastic steps.

        Each step is written through the posterior q(x_{k-1} | x_k, x0_hat), where x0_hat is
        the clean image predicted from eps. Without clipping this is exactly the update
        x_{k-1} = (x_k - beta_k / sqrt(1 - abar_k) * eps) / sqrt(alpha_k) + sqrt(beta_k) z.
        With the cosine schedule, alpha_k ~ 0.001 at the first step: that update multiplies
        the network's error by ~30 and the sample blows up. Clipping x0_hat to the data range
        (``clip=1.0`` for images in [-1, 1]) keeps it stable, as in Ho et al. (2020).

        Returns the final samples and, if ``keep`` is given, ``keep`` intermediate states
        from noise to data (for plots and GIFs).
        """
        x = torch.randn(shape, device=device)
        traj = [x.cpu()]
        save_at = _save_points(self.T, keep)
        for i, k in enumerate(reversed(range(self.T))):
            kk = torch.full((shape[0],), k, device=device)
            eps = model(x, self._t(kk))
            beta, alpha, abar = self.betas[k], self.alphas[k], self.abar[k]
            abar_prev = self.abar[k - 1] if k > 0 else torch.tensor(1.0)
            x0 = (x - (1 - abar).sqrt() * eps) / abar.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
            x = (abar_prev.sqrt() * beta * x0 + alpha.sqrt() * (1 - abar_prev) * x) / (1 - abar)
            if k > 0:
                x = x + beta.sqrt() * torch.randn_like(x)
            if i + 1 in save_at:
                traj.append(x.cpu())
        return (x, traj) if keep else x

    @torch.no_grad()
    def sample_ddim(self, model, shape, steps: int = 50, device="cpu", keep=None, clip=None):
        """DDIM (Song et al. 2021) with eta = 0: deterministic, follows the probability-flow ODE.

        Uses the same network as DDPM, but only ``steps`` evaluations.
        """
        ks = torch.linspace(self.T - 1, 0, steps).round().long()
        x = torch.randn(shape, device=device)
        traj = [x.cpu()]
        save_at = _save_points(steps, keep)
        for i, k in enumerate(ks):
            kk = torch.full((shape[0],), int(k), device=device)
            eps = model(x, self._t(kk))
            abar = self.abar[k]
            abar_prev = self.abar[ks[i + 1]] if i + 1 < steps else torch.tensor(1.0)
            x0 = (x - (1 - abar).sqrt() * eps) / abar.sqrt()
            if clip is not None:
                x0 = x0.clamp(-clip, clip)
                eps = (x - abar.sqrt() * x0) / (1 - abar).sqrt()
            x = abar_prev.sqrt() * x0 + (1 - abar_prev).sqrt() * eps
            if i + 1 in save_at:
                traj.append(x.cpu())
        return (x, traj) if keep else x


def _save_points(n_steps: int, keep):
    if not keep:
        return set()
    return {round(n_steps * j / keep) for j in range(1, keep + 1)}

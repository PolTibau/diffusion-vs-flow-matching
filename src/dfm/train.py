"""A single training loop shared by both methods: only ``method.loss`` differs."""

import copy
import itertools

import torch
from tqdm.auto import tqdm


def train(model, method, loader, steps: int, lr: float = 1e-3, device: str = "cpu",
          ema_decay: float | None = None, grad_clip: float = 1.0):
    """Trains ``model`` for ``steps`` optimisation steps.

    Returns ``(model, losses)``; with ``ema_decay`` the returned model is the exponential
    moving average of the weights, which gives noticeably cleaner samples on images.
    """
    model.to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False) if ema_decay else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    losses = []
    batches = itertools.cycle(loader)
    for _ in (bar := tqdm(range(steps))):
        (x,) = next(batches)
        loss = method.loss(model, x.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        if ema is not None:
            # warm-up: a small decay at first, so the average does not stay stuck at the init
            decay = min(ema_decay, (1 + len(losses)) / (10 + len(losses)))
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.lerp_(p, 1 - decay)
        losses.append(loss.item())
        if len(losses) % 50 == 0:
            bar.set_postfix(loss=f"{sum(losses[-50:]) / 50:.4f}")
    model.eval()
    return (ema if ema is not None else model), losses

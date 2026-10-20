"""Conditional flow matching (Lipman et al. 2023; Liu et al. 2023, rectified flow).

Convention: t = 0 is noise and t = 1 is data. Along the straight path

    x_t = (1 - t) * x_0 + t * x_1,    x_0 ~ N(0, I),  x_1 ~ data,

the conditional velocity is x_1 - x_0, and the network learns v(x_t, t) by regression.
Sampling integrates the ODE dx/dt = v(x, t) from t = 0 to t = 1.

With ``ot=True`` noise and data in each minibatch are paired by optimal transport
(Tong et al. 2023, OT-CFM): the learned paths get straighter and need fewer steps.
"""

import torch
import torch.nn.functional as F


class FlowMatching:
    def __init__(self, ot: bool = False):
        self.ot = ot

    @staticmethod
    def _bc(v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return v.view(-1, *([1] * (x.dim() - 1)))

    @staticmethod
    def ot_pairing(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Reorders x0 so that (x0[i], x1[i]) solves the minibatch OT problem (squared cost)."""
        from scipy.optimize import linear_sum_assignment

        cost = torch.cdist(x0.flatten(1), x1.flatten(1)) ** 2
        rows, cols = linear_sum_assignment(cost.cpu().numpy())
        perm = torch.empty_like(torch.as_tensor(cols))
        perm[torch.as_tensor(cols)] = torch.as_tensor(rows)
        return x0[perm.to(x0.device)]

    def loss(self, model, x1: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn_like(x1)
        if self.ot:
            x0 = self.ot_pairing(x0, x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        tb = self._bc(t, x1)
        xt = (1 - tb) * x0 + tb * x1
        return F.mse_loss(model(xt, t), x1 - x0)

    @torch.no_grad()
    def sample(self, model, shape, steps: int = 50, method: str = "euler", device="cpu", keep=None, x0=None):
        """Integrates the learned ODE with Euler or midpoint (RK2) steps."""
        x = torch.randn(shape, device=device) if x0 is None else x0.to(device)
        traj = [x.cpu()]
        save_at = _save_points(steps, keep)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((shape[0],), i * dt, device=device)
            if method == "euler":
                x = x + dt * model(x, t)
            elif method == "midpoint":
                x_mid = x + 0.5 * dt * model(x, t)
                x = x + dt * model(x_mid, t + 0.5 * dt)
            else:
                raise ValueError(f"unknown method {method!r}")
            if i + 1 in save_at:
                traj.append(x.cpu())
        return (x, traj) if keep else x


def _save_points(n_steps: int, keep):
    if not keep:
        return set()
    return {round(n_steps * j / keep) for j in range(1, keep + 1)}

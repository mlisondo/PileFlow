"""Target conditional flow matching (Lipman et al. 2022), as used in FlowSim.

Probability path (FlowSim eq. 4-5), with x0 ~ N(0, I) and x1 the data:

    z_t = t x1 + (1 - (1 - sigma_min) t) x0
    u_t = x1 - (1 - sigma_min) x0

Training times follow FlowSim's `alpha` option: t = u^(1/(1+alpha)),
u ~ U(0,1), i.e. density (1+alpha) t^alpha, emphasising t -> 1.
"""
from __future__ import annotations

import torch


class TargetConditionalFlowMatching:
    def __init__(self, sigma_min: float = 1e-4, alpha: float = 1.0):
        if not 0.0 <= sigma_min < 1.0:
            raise ValueError("sigma_min must be in [0, 1)")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.sigma_min = float(sigma_min)
        self.alpha = float(alpha)

    # training
    def sample_training_tuple(self, x1: torch.Tensor, generator: torch.Generator | None = None):
        n = x1.shape[0]
        u = torch.rand(n, device=x1.device, dtype=x1.dtype, generator=generator)
        t = u.pow(1.0 / (1.0 + self.alpha))
        x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
        tt = t.view(-1, *([1] * (x1.ndim - 1)))
        zt = tt * x1 + (1.0 - (1.0 - self.sigma_min) * tt) * x0
        ut = x1 - (1.0 - self.sigma_min) * x0
        return t, zt, ut

    def endpoint_from_velocity(self, t: torch.Tensor, zt: torch.Tensor, v: torch.Tensor):
        """x1 implied by a velocity at (t, z_t): x1 = (1-s) z_t + (1 - (1-s) t) v."""
        a = 1.0 - self.sigma_min
        tt = t.view(-1, *([1] * (zt.ndim - 1)))
        return a * zt + (1.0 - a * tt) * v

    # sampling
    @torch.no_grad()
    def sample(self, model, neutral_all, charged_pu, charged_lv, solver: str = "dopri5",
               steps: int = 50, atol: float = 1e-5, rtol: float = 1e-5,
               generator: torch.Generator | None = None, return_nfe: bool = False,
               z0: torch.Tensor | None = None):
        """Integrate dz/dt = v(t, z | c) from t=0 (noise) to t=1.

        solver: 'euler' | 'midpoint' | 'rk4' (fixed `steps`) or 'dopri5'
        (adaptive, torchdiffeq). The initial noise is `z0` if given, else drawn
        with `generator`.
        """
        n = neutral_all.shape[0]
        dev, dt_ = neutral_all.device, neutral_all.dtype
        z = z0.to(dev, dt_) if z0 is not None else torch.randn(
            (n, 1, 9, 9), device=dev, dtype=dt_, generator=generator)
        nfe = 0

        def v(t_scalar, state):
            nonlocal nfe
            nfe += 1
            t = torch.full((n,), float(t_scalar), device=dev, dtype=dt_)
            return model(t, state, neutral_all, charged_pu, charged_lv)

        if solver in {"euler", "midpoint", "rk4"}:
            h = 1.0 / steps
            for i in range(steps):
                t0 = i * h
                if solver == "euler":
                    z = z + h * v(t0, z)
                elif solver == "midpoint":
                    z = z + h * v(t0 + h / 2, z + h / 2 * v(t0, z))
                else:
                    k1 = v(t0, z)
                    k2 = v(t0 + h / 2, z + h / 2 * k1)
                    k3 = v(t0 + h / 2, z + h / 2 * k2)
                    k4 = v(t0 + h, z + h * k3)
                    z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        elif solver == "dopri5":
            try:
                from torchdiffeq import odeint
            except ImportError as e:
                raise ImportError("dopri5 needs torchdiffeq (pip install torchdiffeq) "
                                  "or use --ode-solver euler/midpoint/rk4") from e
            times = torch.tensor([0.0, 1.0], device=dev, dtype=dt_)
            z = odeint(v, z, times, method="dopri5", atol=atol, rtol=rtol)[-1]
        else:
            raise ValueError("solver must be one of euler, midpoint, rk4, dopri5")
        return (z, nfe) if return_nfe else z

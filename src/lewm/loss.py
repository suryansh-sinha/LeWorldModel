"""SIGReg (App. A, p. 13) and the LeWM training loss (Algorithm 3, p. 17)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_DIRS = 1024
KNOTS = 17
T_MIN, T_MAX = 0.2, 4.0


def sigreg(z: torch.Tensor, num_dirs: int = NUM_DIRS, generator: torch.Generator | None = None) -> torch.Tensor:
    """Epps-Pulley test against N(0, 1) on random 1-D projections, averaged over directions and time steps.

    z: (T, N, D), one batch of N embeddings per time step -> scalar (FP32).
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()                                                        # (T, N, D)
        A = torch.randn(z.shape[-1], num_dirs, device=z.device, generator=generator)
        A = A / A.norm(dim=0, keepdim=True)                                  # (D, M) unit-norm columns
        h = z @ A                                                            # (T, N, M) projections
        t = torch.linspace(T_MIN, T_MAX, KNOTS, device=z.device)             # (K,)
        x = h.unsqueeze(-1) * t                                              # (T, N, M, K)
        re, im = torch.cos(x).mean(1), torch.sin(x).mean(1)                  # (T, M, K) empirical CF
        gauss = torch.exp(-0.5 * t**2)                                       # (K,) N(0,1) CF, also the weight
        err = ((re - gauss) ** 2 + im**2) * gauss                            # (T, M, K)
        # The integrand is even in t, so the integral over all of R is 2x the [0.2, 4] quadrature.
        # No xN factor (the classical EP test statistic uses N*T); see PROGRESS.md decisions.
        stat = 2.0 * torch.trapezoid(err, t, dim=-1)                         # (T, M)
        return stat.mean()


def lewm_loss(model, frames: torch.Tensor, actions: torch.Tensor, lam: float = 0.1,
              generator: torch.Generator | None = None) -> dict:
    """frames uint8 (B, 4, 3, H, W), actions (B, 4, 10) -> dict(loss, pred_loss, sigreg)."""
    emb, pred = model(frames, actions)                    # (B, 4, D), (B, 3, D)
    pred_loss = F.mse_loss(pred.float(), emb[:, 1:].float())  # no stop-gradient on the target
    sig = sigreg(emb.transpose(0, 1), generator=generator)    # (4, B, D) -> scalar
    return {"loss": pred_loss + lam * sig, "pred_loss": pred_loss, "sigreg": sig}

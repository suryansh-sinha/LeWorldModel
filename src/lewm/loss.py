"""SIGReg (App. A, p. 13) and the LeWM training loss (Algorithm 3, p. 17)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

NUM_DIRS = 1024
KNOTS = 17
T_MIN, T_MAX = 0.2, 4.0


def sigreg(z: torch.Tensor, num_dirs: int = NUM_DIRS, generator: torch.Generator | None = None,
           scale_n: bool = False) -> torch.Tensor:
    """Epps-Pulley test against N(0, 1) on random 1-D projections, averaged over directions and time steps.

    z: (T, N, D), one batch of N embeddings per time step -> scalar (FP32).
    scale_n: multiply by N (the conventional EP statistic N*T); off by default (paper's formula, App. A).
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
        # scale_n=True gives the classical EP statistic N*T. The training config uses it: without it the
        # embedding collapsed to effective rank ~3/192 (see PROGRESS.md, Phase 4a).
        stat = 2.0 * torch.trapezoid(err, t, dim=-1)                         # (T, M)
        return stat.mean() * (z.shape[1] if scale_n else 1.0)


def lewm_loss(model, frames: torch.Tensor, actions: torch.Tensor, lam: float = 0.1,
              generator: torch.Generator | None = None, sigreg_scale_n: bool = False) -> dict:
    """frames uint8 (B, 4, 3, H, W), actions (B, 4, 10) -> dict(loss, pred_loss, sigreg, emb_std)."""
    emb, pred = model(frames, actions)                    # (B, 4, D), (B, 3, D)
    pred_loss = F.mse_loss(pred.float(), emb[:, 1:].float())  # no stop-gradient on the target
    sig = sigreg(emb.transpose(0, 1), generator=generator, scale_n=sigreg_scale_n)  # (4, B, D) -> scalar
    emb_std = emb.detach().float().std(dim=0).mean()          # per-dim std over the batch; collapse -> 0
    return {"loss": pred_loss + lam * sig, "pred_loss": pred_loss, "sigreg": sig, "emb_std": emb_std,
            "emb": emb.detach()}


def effective_rank(z: torch.Tensor) -> float:
    """exp(entropy) of the normalized covariance eigenvalues of z (N, D); D = isotropic, 1 = one direction."""
    ev = torch.linalg.eigvalsh(torch.cov(z.double().T)).clamp_min(0)
    p = ev / ev.sum()
    return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()

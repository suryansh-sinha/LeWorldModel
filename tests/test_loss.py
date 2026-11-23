import math

import torch

from lewm.loss import KNOTS, T_MAX, T_MIN, lewm_loss, sigreg
from lewm.model import LeWM


def _sig(z):
    return sigreg(z, generator=torch.Generator().manual_seed(0)).item()


def test_sigreg_values():
    g = torch.Generator().manual_seed(1)
    gauss = _sig(torch.randn(4, 1024, 192, generator=g))
    zeros = _sig(torch.zeros(4, 1024, 192))
    const = _sig(torch.full((4, 1024, 192), 0.7))
    shifted = _sig(torch.randn(4, 1024, 192, generator=g) + 3.0)
    print(f"\nsigreg: gauss {gauss:.2e}, zeros {zeros:.2e}, const {const:.2e}, shifted {shifted:.2e}")
    assert gauss < 2e-3  # sampling-noise floor, ~1/N (about 1e-3 at N = 1024 with the x2)
    for bad in (zeros, const, shifted):
        assert bad > 10 * gauss

    # Exact value for a point mass at 0: ECF = 1, so err(t) = (1 - e^{-t^2/2})^2 * e^{-t^2/2},
    # integrated over both halves of R (x2).
    t = [T_MIN + i * (T_MAX - T_MIN) / (KNOTS - 1) for i in range(KNOTS)]
    f = [(1 - math.exp(-s * s / 2)) ** 2 * math.exp(-s * s / 2) for s in t]
    expected = 2 * sum((f[i] + f[i + 1]) / 2 * (t[i + 1] - t[i]) for i in range(KNOTS - 1))
    assert math.isclose(zeros, expected, rel_tol=1e-5)


def test_sigreg_fp32_under_autocast():
    z = torch.randn(4, 64, 192).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert sigreg(z).dtype == torch.float32


def _batch(B=4):
    torch.manual_seed(0)
    return torch.randint(0, 256, (B, 4, 3, 224, 224), dtype=torch.uint8), torch.randn(B, 4, 10)


def test_gradients_flow():
    torch.manual_seed(0)
    model = LeWM().train()
    frames, actions = _batch()
    out = lewm_loss(model, frames, actions)
    out["loss"].backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    # adaLN-Zero: predictor attn/MLP and the action encoder get zero grads at step 0, by design.
    assert all(p.grad.abs().sum() > 0 for p in model.encoder.parameters() if p.dim() > 1)
    assert all(blk.ada.weight.grad.abs().sum() > 0 for blk in model.predictor.blocks)

    model.zero_grad()
    sigreg(model.encode(frames).transpose(0, 1)).backward()
    assert model.encoder.patch_embed.weight.grad.abs().sum() > 0


def test_a3_does_not_change_loss():
    torch.manual_seed(0)
    model = LeWM().train()
    frames, actions = _batch()
    actions2 = actions.clone()
    actions2[:, 3] += torch.randn(4, 10) * 5
    losses = []
    for a in (actions, actions2):
        torch.manual_seed(123)  # same dropout masks and SIGReg directions
        with torch.no_grad():
            losses.append(lewm_loss(model, frames, a)["loss"])
    assert torch.equal(losses[0], losses[1])

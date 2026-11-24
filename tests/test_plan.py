import torch

from lewm.plan import cem, rollout_cost


def test_cem_converges_on_quadratic():
    torch.manual_seed(0)
    target = torch.rand(5, 10) - 0.5                                   # known optimum inside the bounds
    lo, hi = torch.full((10,), -2.0), torch.full((10,), 2.0)
    cost = lambda x: ((x - target) ** 2).sum(dim=(1, 2))
    mu = cem(cost, lo, hi, generator=torch.Generator().manual_seed(0))
    assert (mu - target).abs().max() < 0.05


def test_cem_respects_bounds():
    lo, hi = torch.full((10,), -0.3), torch.full((10,), 0.2)
    mu = cem(lambda x: -(x**2).sum(dim=(1, 2)), lo, hi, generator=torch.Generator().manual_seed(0))
    assert (mu >= lo - 1e-6).all() and (mu <= hi + 1e-6).all()


def test_rollout_context_window():
    D, n, H = 4, 7, 5
    calls = []

    def fake_predict(z, a):                                            # records inputs; "next" = last + 1
        calls.append((z.clone(), a.clone()))
        return z + 1.0

    ctx_z = torch.stack([torch.full((D,), v) for v in (10.0, 20.0)])    # 2 real frames
    ctx_a = torch.full((1, 10), -1.0)                                  # executed block after the older one
    cands = torch.arange(n * H, dtype=torch.float32).view(n, H, 1).expand(n, H, 10)
    cost = rollout_cost(fake_predict, ctx_z, ctx_a, torch.zeros(D), cands)

    assert cost.shape == (n,) and len(calls) == H
    for h, (z, a) in enumerate(calls):
        assert z.shape[1] == a.shape[1] == min(2 + h, 3)               # window never exceeds 3
    z0, a0 = calls[0]
    assert torch.equal(z0[0, :, 0], torch.tensor([10.0, 20.0]))        # real context first
    assert torch.equal(a0[0, :, 0], torch.tensor([-1.0, cands[0, 0, 0]]))
    z2, a2 = calls[2]                                                  # window slid: real, pred, pred
    assert torch.equal(z2[0, :, 0], torch.tensor([20.0, 21.0, 22.0]))
    assert torch.equal(a2[0, :, 0], cands[0, :3, 0])
    assert torch.allclose(cost, torch.full((n,), D * 25.0**2))         # z_hat_5 = 20 + 5 = 25

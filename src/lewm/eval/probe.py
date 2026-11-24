"""Linear and MLP probes on frozen LeWM embeddings (Table 1, p. 8).

Usage:
    uv run python -m lewm.eval.probe --ckpt outputs/runs/pusht/last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.data.pusht import EPS_PER_GROUP, FRAMESKIP, load_split
from lewm.loss import effective_rank
from lewm.model import LeWM

PROPERTIES = {"agent": slice(0, 2), "block": slice(2, 4), "angle": slice(4, 5)}
N_PROBE_TRAIN_GROUPS = 14  # of the 19 val groups; the other 5 are the probe test set


@torch.no_grad()
def embed_groups(model: LeWM, h5_path: str, groups, device, batch: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Every 5th frame of every episode in `groups` -> z (N, 192) float32 on CPU, state (N, 7)."""
    zs, states, buf = [], [], []

    def flush():
        x = torch.from_numpy(np.concatenate(buf)).to(device).permute(0, 3, 1, 2)  # (n, 3, 224, 224) uint8
        with torch.autocast("cuda", dtype=torch.bfloat16):
            zs.append(model.encode(x[:, None]).float()[:, 0].cpu())                  # (n, 192)
        buf.clear()

    with h5py.File(h5_path, "r") as f:
        off, lens = f["ep_offset"][:], f["ep_len"][:]
        for g in groups:
            for e in range(g * EPS_PER_GROUP, (g + 1) * EPS_PER_GROUP):
                rows = slice(off[e], off[e] + lens[e])
                buf.append(f["pixels"][rows][::FRAMESKIP])     # contiguous read, then subsample
                states.append(f["state"][rows][::FRAMESKIP])
                if sum(len(b) for b in buf) >= batch:
                    flush()
        if buf:
            flush()
    return torch.cat(zs), torch.from_numpy(np.concatenate(states))


def pearson_r(pred: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson r per output dim, averaged."""
    p, t = pred - pred.mean(0), y - y.mean(0)
    return ((p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0))).mean().item()


def linear_probe(xtr, ytr, xte) -> torch.Tensor:
    A = torch.cat([xtr, torch.ones(len(xtr), 1)], 1)                   # (N, 193) with bias column
    W = torch.linalg.lstsq(A.double(), ytr.double()).solution.float()   # (193, out)
    return torch.cat([xte, torch.ones(len(xte), 1)], 1) @ W


def mlp_probe(xtr, ytr, xte, device, epochs: int = 100, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.GELU(), nn.Linear(256, ytr.shape[1])).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    xtr, ytr = xtr.to(device), ytr.to(device)
    for _ in range(epochs):
        for idx in torch.randperm(len(xtr), device=device).split(1024):
            loss = F.mse_loss(net(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net(xte.to(device)).cpu()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/runs/pusht/last.pt")
    p.add_argument("--h5", default="data/pusht_expert_train.h5")
    p.add_argument("--out", default="outputs/eval")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    val_groups = load_split()["val_groups"]                   # sorted
    tr_groups, te_groups = val_groups[:N_PROBE_TRAIN_GROUPS], val_groups[N_PROBE_TRAIN_GROUPS:]
    cache = out / "probe_cache.pt"
    if cache.exists():
        d = torch.load(cache)
        assert d["ckpt"] == args.ckpt and d["tr_groups"] == tr_groups
    else:
        model = LeWM().to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
        model.eval()
        ztr, str_ = embed_groups(model, args.h5, tr_groups, device)
        zte, ste = embed_groups(model, args.h5, te_groups, device)
        d = {"ckpt": args.ckpt, "tr_groups": tr_groups, "ztr": ztr, "str": str_, "zte": zte, "ste": ste}
        torch.save(d, cache)
    ztr, zte = d["ztr"], d["zte"]
    print(f"probe-train frames {len(ztr):,} ({len(tr_groups)} groups), probe-test frames {len(zte):,} ({len(te_groups)} groups)")
    print(f"effective rank of z (probe-train frames): {effective_rank(ztr):.1f} / {ztr.shape[1]}")

    # Two protocols on the same val-group frames:
    #  - "random": 75/25 random frame split (comparable to Table 1; the paper doesn't state its split)
    #  - "group":  train on 14 val groups, test on the other 5 (unseen starting states; stricter)
    z_all, s_all = torch.cat([ztr, zte]), torch.cat([d["str"], d["ste"]]).float()
    perm = torch.randperm(len(z_all), generator=torch.Generator().manual_seed(0))
    n_tr = int(0.75 * len(z_all))
    splits = {
        "random": (z_all[perm[:n_tr]], s_all[perm[:n_tr]], z_all[perm[n_tr:]], s_all[perm[n_tr:]]),
        "group": (ztr, d["str"].float(), zte, d["ste"].float()),
    }
    results = {"effective_rank": effective_rank(ztr)}
    for split, (xa, sa, xb, sb) in splits.items():
        results[split] = {}
        print(f"\n[{split} split] train {len(xa):,} / test {len(xb):,} frames")
        print(f"{'Property':8s} | {'Linear MSE':>10s} {'r':>6s} | {'MLP MSE':>8s} {'r':>6s}")
        for name, sl in PROPERTIES.items():
            ytr, yte = sa[:, sl], sb[:, sl]
            mu, sd = ytr.mean(0), ytr.std(0)                    # z-score labels with probe-train stats
            ytr, yte = (ytr - mu) / sd, (yte - mu) / sd
            res = {}
            for kind, pred in (("linear", linear_probe(xa, ytr, xb)), ("mlp", mlp_probe(xa, ytr, xb, device))):
                res[kind] = {"mse": F.mse_loss(pred, yte).item(), "r": pearson_r(pred, yte)}
            results[split][name] = res
            print(f"{name:8s} | {res['linear']['mse']:10.4f} {res['linear']['r']:6.3f} | "
                  f"{res['mlp']['mse']:8.4f} {res['mlp']['r']:6.3f}")
    (out / "probes.json").write_text(json.dumps(results, indent=1))
    print(f"\n-> {out / 'probes.json'}")


if __name__ == "__main__":
    main()

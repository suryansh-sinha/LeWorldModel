"""LeWM training loop.

Usage:
    uv run python -m lewm.train --config configs/smoke.yaml
    uv run python -m lewm.train --config configs/pusht.yaml [--max_steps 50] [--resume]
"""

from __future__ import annotations

import argparse
import math
import shutil
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from lewm.data.pusht import PushTWindows, episodes_of_groups, load_split
from lewm.loss import lewm_loss
from lewm.model import LeWM

LOG_EVERY = 25


def action_stats(h5_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension mean/std of all action rows in train-group episodes -> (2,), (2,)."""
    with h5py.File(h5_path, "r") as f:
        off, lens, act = f["ep_offset"][:], f["ep_len"][:], f["action"][:]
    eps = episodes_of_groups(load_split()["train_groups"])
    rows = np.concatenate([np.arange(off[e], off[e] + lens[e]) for e in eps])
    a = act[rows].astype(np.float64)
    return torch.tensor(a.mean(0), dtype=torch.float32), torch.tensor(a.std(0), dtype=torch.float32)


def normalize_actions(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    B, T, _ = actions.shape                                   # (B, 4, 10) = 4 blocks of 5 steps x 2 dims
    a = (actions.view(B, T, -1, 2) - mean) / std              # (B, 4, 5, 2)
    return a.view(B, T, -1)                                   # (B, 4, 10)


def lr_lambda(total_steps: int, warmup_frac: float):
    warmup = max(1, int(warmup_frac * total_steps))

    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return f


@torch.no_grad()
def evaluate(model, loader, mean, std, lam, device) -> dict:
    model.eval()
    sums = {"pred_loss": 0.0, "sigreg": 0.0, "emb_std": 0.0, "loss": 0.0}
    n = 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        actions = normalize_actions(batch["actions"].to(device, non_blocking=True), mean, std)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = lewm_loss(model, frames, actions, lam=lam)
        for k in sums:
            sums[k] += out[k].item()
        n += 1
    model.train()
    return {f"val/{k}": v / n for k, v in sums.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--max_steps", type=int, default=None, help="stop early (timing check)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--stride", type=int, default=None, help="override the config's window stride")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.stride is not None:
        cfg["stride"] = args.stride
    assert cfg["stride"] in (1, 5), "set stride (decision 9) in the config or with --stride"

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda")
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- data
    train_ds = PushTWindows(cfg["h5_path"], "train", stride=cfg["stride"])
    if cfg.get("subset"):
        idx = np.random.default_rng(cfg["seed"]).choice(len(train_ds), cfg["subset"], replace=False)
        train_ds = Subset(train_ds, idx.tolist())
    val_full = PushTWindows(cfg["h5_path"], "val", stride=cfg["stride"])
    val_idx = np.random.default_rng(cfg["seed"] + 1).permutation(len(val_full))[: cfg["val_batches"] * cfg["batch_size"]]
    val_ds = Subset(val_full, val_idx.tolist())

    loader_kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True,
                     persistent_workers=True, prefetch_factor=4, drop_last=True)
    gen = torch.Generator()
    train_loader = DataLoader(train_ds, shuffle=True, generator=gen, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg["epochs"]

    mean, std = action_stats(cfg["h5_path"])
    print(f"train windows {len(train_ds):,} ({steps_per_epoch:,} steps/epoch, {total_steps:,} total), "
          f"val windows {len(val_ds):,}; action mean {mean.tolist()} std {std.tolist()}")
    mean, std = mean.to(device), std.to(device)

    # ---- model / optim
    model = LeWM().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total_steps, cfg["warmup_frac"]))
    start_epoch, step, wandb_id = 0, 0, None
    if args.resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, step, wandb_id = ck["epoch"] + 1, ck["step"], ck.get("wandb_id")
        print(f"resumed from {out_dir / 'last.pt'}: next epoch {start_epoch}, step {step}")

    import wandb

    run = wandb.init(project=cfg["wandb"]["project"], mode=cfg["wandb"]["mode"], name=out_dir.name,
                     config=cfg, dir=str(out_dir), id=wandb_id, resume="allow")

    # ---- train
    model.train()
    torch.cuda.reset_peak_memory_stats()
    done = False
    t_last, wait = time.perf_counter(), 0.0  # logging window; NOT reset per epoch (a window can span epochs)
    for epoch in range(start_epoch, cfg["epochs"]):
        gen.manual_seed(cfg["seed"] + epoch)  # same data order per epoch, also after resume
        t_fetch = time.perf_counter()
        steady_t0 = None
        for i, batch in enumerate(train_loader):
            wait += time.perf_counter() - t_fetch
            frames = batch["frames"].to(device, non_blocking=True)                  # (B, 4, 3, 224, 224) uint8
            actions = normalize_actions(batch["actions"].to(device, non_blocking=True), mean, std)  # (B, 4, 10)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = lewm_loss(model, frames, actions, lam=cfg["lam"])
            opt.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            if args.max_steps:
                torch.cuda.synchronize()  # timing check only: makes data_wait_frac exclude async GPU work

            if args.max_steps and i == 10:
                torch.cuda.synchronize()
                steady_t0, wait_at_t0 = time.perf_counter(), wait
            if step % LOG_EVERY == 0:
                now = time.perf_counter()
                metrics = {f"train/{k}": v.item() for k, v in out.items()}
                metrics.update({"train/grad_norm": grad_norm.item(), "lr": sched.get_last_lr()[0],
                                "it_per_s": LOG_EVERY / (now - t_last), "data_wait_frac": wait / (now - t_last),
                                "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30, "epoch": epoch})
                run.log(metrics, step=step)
                print(f"ep {epoch} step {step}/{total_steps} loss {metrics['train/loss']:.4f} "
                      f"pred {metrics['train/pred_loss']:.4f} sigreg {metrics['train/sigreg']:.4f} "
                      f"emb_std {metrics['train/emb_std']:.3f} gnorm {metrics['train/grad_norm']:.2f} "
                      f"lr {metrics['lr']:.2e} {metrics['it_per_s']:.2f} it/s wait {metrics['data_wait_frac']:.0%}",
                      flush=True)
                t_last, wait = now, 0.0
            if args.max_steps and step >= args.max_steps:
                torch.cuda.synchronize()
                dt = time.perf_counter() - steady_t0
                n = i - 10
                print(f"\nTIMING over steps 11-{i + 1}: {n / dt:.2f} it/s, "
                      f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
                for s, spe in ((1, 13_845), (5, 2_824)):
                    print(f"  stride {s}: {spe:,} steps/epoch -> 10 epochs ~ {10 * spe / (n / dt) / 3600:.1f} h")
                done = True
                break
            t_fetch = time.perf_counter()
        if done:
            break

        val = evaluate(model, val_loader, mean, std, cfg["lam"], device)
        run.log(val, step=step)
        print(f"== epoch {epoch} done, step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in val.items()), flush=True)
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": epoch,
              "step": step, "action_mean": mean.cpu(), "action_std": std.cpu(), "config": cfg, "wandb_id": run.id}
        if cfg.get("keep_epoch_ckpts", True):
            torch.save(ck, out_dir / f"epoch_{epoch}.pt")
            shutil.copyfile(out_dir / f"epoch_{epoch}.pt", out_dir / "last.pt")
        else:
            torch.save(ck, out_dir / "last.pt")
    run.finish()


if __name__ == "__main__":
    main()

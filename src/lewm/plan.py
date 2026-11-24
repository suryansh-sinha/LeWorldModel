"""CEM planning with LeWM in swm/PushT-v1 (Alg. 2, p. 14; App. D, p. 18).

Usage:
    uv run python -m lewm.plan --ckpt outputs/runs/pusht_scaleN/last.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch

from lewm.data.pusht import FRAMESKIP, episodes_of_groups, load_split

HORIZON = 5          # action blocks per plan (25 env steps)
BLOCK = FRAMESKIP * 2  # one block = 5 steps x 2 dims = 10 numbers
N_SAMPLES, N_ELITES, N_ITERS = 300, 30, 30
CONTEXT = 3          # predictor history length
GOAL_OFFSET = 25     # goal is 25 env steps after the start
BUDGET = 50          # env steps per episode
PAIRS_PATHS = {"val": Path("configs/eval/pusht_eval_pairs.json"),            # val groups only (held out)
               "random": Path("configs/eval/pusht_eval_pairs_random.json")}  # any episode, train or val


def cem(cost_fn, lo: torch.Tensor, hi: torch.Tensor, H: int = HORIZON, dim: int = BLOCK, n: int = N_SAMPLES,
        k: int = N_ELITES, iters: int = N_ITERS, generator: torch.Generator | None = None) -> torch.Tensor:
    """Cross-entropy method over action sequences (H, dim) in normalized units.

    cost_fn: (n, H, dim) -> (n,). lo, hi: (dim,) bounds; samples are clipped to them so the cost
    is evaluated on exactly the actions that would be executed. Returns the final mean (H, dim).
    """
    mu = torch.zeros(H, dim, device=lo.device)
    var = torch.ones(H, dim, device=lo.device)
    for _ in range(iters):
        eps = torch.randn(n, H, dim, device=lo.device, generator=generator)
        samples = torch.maximum(torch.minimum(mu + var.sqrt() * eps, hi), lo)   # (n, H, dim)
        elites = samples[cost_fn(samples).topk(k, largest=False).indices]       # (k, H, dim)
        mu, var = elites.mean(0), elites.var(0, unbiased=False)
    return mu


def rollout_cost(predict, ctx_z: torch.Tensor, ctx_a: torch.Tensor, z_goal: torch.Tensor,
                 candidates: torch.Tensor) -> torch.Tensor:
    """Autoregressive latent rollout; cost = ||z_hat_H - z_goal||^2.

    predict: (n, T<=3, D), (n, T, 10) -> (n, T, D)  (LeWM.predict)
    ctx_z: (c, D) embeddings of the last c <= 3 real frames, oldest first
    ctx_a: (c-1, 10) normalized action blocks executed after each of the older real frames
    candidates: (n, H, 10) normalized action blocks; block 0 follows the newest real frame
    """
    n, H, _ = candidates.shape
    z_seq = ctx_z.unsqueeze(0).expand(n, -1, -1)                                   # (n, c, D)
    a_seq = torch.cat([ctx_a.unsqueeze(0).expand(n, -1, -1), candidates[:, :1]], 1)  # (n, c, 10)
    for h in range(H):
        z_next = predict(z_seq[:, -CONTEXT:], a_seq[:, -CONTEXT:])[:, -1]           # (n, D)
        if h < H - 1:
            z_seq = torch.cat([z_seq, z_next[:, None]], 1)
            a_seq = torch.cat([a_seq, candidates[:, h + 1 : h + 2]], 1)
    return ((z_next.float() - z_goal.float()) ** 2).sum(-1)                        # (n,)


def make_eval_pairs(h5_path: str, n: int = 50, seed: int = 0, split: str = "val") -> list[dict]:
    with h5py.File(h5_path, "r") as f:
        off, lens = f["ep_offset"][:], f["ep_len"][:]
    rng = np.random.default_rng(seed)
    pool = episodes_of_groups(load_split()["val_groups"]) if split == "val" else np.arange(len(lens))
    eps = rng.choice(pool, n, replace=False)
    pairs = []
    for e in sorted(int(x) for x in eps):
        t = int(rng.integers(0, lens[e] - GOAL_OFFSET))      # t + 25 <= L - 1
        pairs.append({"episode": e, "t": t, "row": int(off[e] + t)})
    return pairs


def load_or_make_pairs(h5_path: str, split: str = "val") -> list[dict]:
    path = PAIRS_PATHS[split]
    if path.exists():
        return json.loads(path.read_text())["pairs"]
    pairs = make_eval_pairs(h5_path, split=split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": 0, "goal_offset": GOAL_OFFSET, "split": "val_groups" if split == "val" else "all_episodes",
                                      "pairs": pairs}, indent=1) + "\n")
    return pairs


def run_episode(env, model, z_of, start_state, goal_state, goal_frame, mean, std, lo, hi, seed, device):
    """MPC: plan 5 blocks, execute all 25 steps, replan, until success or the 50-step budget."""
    u = env.unwrapped
    env.reset(options={"state": start_state, "goal_state": goal_state})
    z_goal = z_of(goal_frame)
    frames, plan_times = [u.render()], []
    hist_z, hist_a = [z_of(frames[-1])], []          # real frames every 5 steps, executed blocks
    gen = torch.Generator(device=device).manual_seed(seed)
    steps, success = 0, False
    predict = lambda z, a: model.predict(z, a)
    while steps < BUDGET and not success:
        ctx_z = torch.stack(hist_z[-CONTEXT:])                                       # (c, D)
        ctx_a = torch.stack(hist_a[-(len(ctx_z) - 1):]) if len(ctx_z) > 1 else torch.zeros(0, BLOCK, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            plan = cem(lambda c: rollout_cost(predict, ctx_z, ctx_a, z_goal, c), lo, hi, generator=gen)
        torch.cuda.synchronize()
        plan_times.append(time.perf_counter() - t0)
        raw = (plan.view(HORIZON, FRAMESKIP, 2) * std + mean).clamp(-1.0, 1.0)       # (5, 5, 2) env units
        for block in raw:
            for a in block.cpu().numpy():
                _, _, terminated, _, _ = env.step(a.astype(np.float32))
                frames.append(u.render())
                steps += 1
                success = success or bool(terminated)
                if success or steps >= BUDGET:
                    break
            if success or steps >= BUDGET:
                break
            hist_z.append(z_of(frames[-1]))
            hist_a.append(((block - mean) / std).reshape(-1))                        # executed, normalized
    final = u._get_obs()                                                               # (7,) final state
    return success, steps, plan_times, frames, final


def save_gif(path: Path, frames, goal_frame) -> None:
    import imageio.v3 as iio

    sep = np.full((frames[0].shape[0], 4, 3), 255, np.uint8)
    iio.imwrite(path, [np.concatenate([f, sep, goal_frame], 1) for f in frames], duration=100, loop=0)


def main() -> None:
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)

    from lewm.model import LeWM

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/runs/pusht_scaleN/last.pt")
    p.add_argument("--h5", default="data/pusht_expert_train.h5")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--split", choices=list(PAIRS_PATHS), default="val", help="pairs from val groups or from any episode")
    p.add_argument("--n_pairs", type=int, default=None, help="evaluate only the first N pairs (debugging)")
    args = p.parse_args()
    out = Path(args.out)
    (out / "gifs").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LeWM().to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    mean, std = ck["action_mean"].to(device), ck["action_std"].to(device)              # (2,), (2,)
    lo = ((-1.0 - mean) / std).repeat(FRAMESKIP)                                       # (10,) normalized bounds
    hi = ((1.0 - mean) / std).repeat(FRAMESKIP)

    @torch.no_grad()
    def z_of(frame: np.ndarray) -> torch.Tensor:                                       # (224, 224, 3) -> (D,)
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(device).permute(2, 0, 1)[None, None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.encode(x)[0, 0]

    pairs = load_or_make_pairs(args.h5, args.split)[: args.n_pairs]
    env = gym.make("swm/PushT-v1")
    f = h5py.File(args.h5, "r")
    results, gifs = [], {"success": 0, "failure": 0}
    for i, pr in enumerate(pairs):
        r = pr["row"]
        start, goal = f["state"][r].astype(np.float64), f["state"][r + GOAL_OFFSET].astype(np.float64)
        goal_frame = f["pixels"][r + GOAL_OFFSET]
        trivial = bool(env.unwrapped.eval_state(goal, start)[0])                         # start already "at goal"?
        ok, steps, times, frames, final = run_episode(env, model, z_of, start, goal, goal_frame, mean, std, lo, hi,
                                               seed=i, device=device)
        ang = abs(float(final[4] - goal[4])) % (2 * math.pi)
        results.append({**pr, "success": ok, "steps": steps, "start_already_success": trivial, "plan_times": times,
                        "final_pos_err": float(np.linalg.norm(final[:4] - goal[:4])),   # success needs < 20
                        "final_agent_err": float(np.linalg.norm(final[:2] - goal[:2])),
                        "final_block_err": float(np.linalg.norm(final[2:4] - goal[2:4])),
                        "final_angle_err": min(ang, 2 * math.pi - ang)})                 # success needs < pi/9
        kind = "success" if ok else "failure"
        if gifs[kind] < 2:
            save_gif(out / "gifs" / f"{kind}_ep{pr['episode']}_t{pr['t']}.gif", frames, goal_frame)
            gifs[kind] += 1
        print(f"[{i + 1:2d}/{len(pairs)}] ep {pr['episode']:5d} t {pr['t']:3d}: {'SUCCESS' if ok else 'fail   '} "
              f"after {steps:2d} steps, plan time {np.mean(times):.2f}s", flush=True)

    n_ok = sum(r["success"] for r in results)
    rate = n_ok / len(results)
    half = 1.96 * math.sqrt(rate * (1 - rate) / len(results))                         # normal-approx 95% CI
    times = np.array([t for r in results for t in r["plan_times"]])
    n_triv = sum(r["start_already_success"] for r in results)
    summary = {"ckpt": args.ckpt, "split": args.split, "n": len(results), "successes": n_ok, "success_rate": rate,
               "start_already_success": n_triv,
               "ci95": [max(0.0, rate - half), min(1.0, rate + half)],
               "plan_time_mean_s": float(times.mean()), "plan_time_std_s": float(times.std())}
    (out / "plan_results.json").write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print(f"\nsuccess {n_ok}/{len(results)} = {rate:.1%} (95% CI {summary['ci95'][0]:.1%}-{summary['ci95'][1]:.1%}); "
          f"plan time {times.mean():.2f} ± {times.std():.2f} s over {len(times)} plans; "
          f"{n_triv} pairs already satisfied the goal at the start -> {out / 'plan_results.json'}")


if __name__ == "__main__":
    main()

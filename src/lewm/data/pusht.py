"""PushT windows: group split, window index, and Dataset.

A window starts at row r0 inside one episode and spans 20 env steps:
  frame k  = row r0 + 5k                       (k = 0..3)
  action k = rows r0 + 5k .. r0 + 5k + 4, flattened to 10 numbers
  state k  = row r0 + 5k

Usage:
    uv run python -m lewm.data.pusht --path data/pusht_expert_train.h5 --out outputs/windows/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the Blosc filter used by `pixels`)
import numpy as np
import torch
from torch.utils.data import Dataset

N_GROUPS = 185          # Total 185 expert demonstrations in dataset
EPS_PER_GROUP = 101     # Each demonstration has 100 more noisy episodes. Total 101
N_VAL_GROUPS = 19       # Val set has 19 of 185 demonstrations. Rest in train set.
SPLIT_SEED = 3072       # Seed for reproducibility
NUM_FRAMES = 4          # Each sample has 4 observations (frames)
FRAMESKIP = 5           # Observations are 5 environment steps apart (5 frames apart)
WINDOW = NUM_FRAMES * FRAMESKIP  # 20 env steps so 20 action vectors i.e. 4,5,2
SPLIT_PATH = Path("configs/splits/pusht_group_split.json")


def make_group_split(n_groups: int = N_GROUPS, n_val: int = N_VAL_GROUPS, seed: int = SPLIT_SEED) -> dict:
    perm = np.random.default_rng(seed).permutation(n_groups)    # Creates shuffled list for each demonstration ID [0,184]
    return {
        "seed": seed,
        "train_groups": sorted(int(g) for g in perm[n_val:]),
        "val_groups": sorted(int(g) for g in perm[:n_val]),
    }


def save_split(split: dict, path: str | Path = SPLIT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1) + "\n")


def load_split(path: str | Path = SPLIT_PATH) -> dict:
    return json.loads(Path(path).read_text())


def episodes_of_groups(groups) -> np.ndarray:
    """Episode ids of the given starting-state groups (group = episode // 101)."""
    return np.concatenate([np.arange(g * EPS_PER_GROUP, (g + 1) * EPS_PER_GROUP) for g in groups])


def build_window_index(ep_offset: np.ndarray, ep_len: np.ndarray, episodes, stride: int = 1) -> np.ndarray:
    """Absolute start rows r0 of all windows inside the given episodes."""
    starts = [ep_offset[e] + np.arange(0, ep_len[e] - WINDOW + 1, stride) for e in episodes]
    return np.concatenate(starts).astype(np.int64)


class PushTWindows(Dataset):
    """Returns dict(frames uint8 (4,3,224,224), actions float32 (4,10), state float32 (4,7), r0)."""

    def __init__(self, h5_path: str | Path, split: str = "train", stride: int = 1, split_path: str | Path = SPLIT_PATH):
        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            ep_offset = f["ep_offset"][:]
            ep_len = f["ep_len"][:]
        assert len(ep_len) == N_GROUPS * EPS_PER_GROUP, f"expected 185x101 episodes, got {len(ep_len)}"
        groups = load_split(split_path)[f"{split}_groups"]
        self.index = build_window_index(ep_offset, ep_len, episodes_of_groups(groups), stride)
        self._f = None  # opened lazily so each DataLoader worker gets its own handle

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        r0 = int(self.index[i])
        rows = [r0 + FRAMESKIP * k for k in range(NUM_FRAMES)]
        frames = torch.from_numpy(self._f["pixels"][rows]).permute(0, 3, 1, 2)
        actions = torch.from_numpy(self._f["action"][r0 : r0 + WINDOW].reshape(NUM_FRAMES, -1))
        state = torch.from_numpy(self._f["state"][rows])
        return {"frames": frames, "actions": actions, "state": state, "r0": r0}


def save_window_grid(ds_by_split: dict, h5_path: str, out_png: Path, per_split: int = 3, seed: int = 0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(h5_path, "r") as f:
        episode_idx, step_idx = f["episode_idx"], f["step_idx"]
        items = []
        rng = np.random.default_rng(seed)
        for name, ds in ds_by_split.items():
            for i in rng.choice(len(ds), per_split, replace=False):
                item = ds[int(i)]
                r0 = item["r0"]
                items.append((name, int(episode_idx[r0]), int(step_idx[r0]), item))

    fig, axes = plt.subplots(len(items), NUM_FRAMES, figsize=(2.4 * NUM_FRAMES, 2.5 * len(items)))
    scale = 224 / 512  # env coordinates are on a 512 px board
    for row, (name, ep, t0, item) in enumerate(items):
        for k in range(NUM_FRAMES):
            ax = axes[row, k]
            ax.imshow(item["frames"][k].permute(1, 2, 0).numpy())
            x, y = item["state"][k, :2].numpy() * scale
            ax.plot(x, y, "r.", markersize=6)
            ax.set_title(f"{name} ep{ep} t={t0 + FRAMESKIP * k}", fontsize=7)
            ax.axis("off")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", default="data/pusht_expert_train.h5")
    p.add_argument("--out", default="outputs/windows/")
    args = p.parse_args()

    split = make_group_split()
    save_split(split)
    print(f"split: {len(split['train_groups'])} train / {len(split['val_groups'])} val groups -> {SPLIT_PATH}")

    with h5py.File(args.path, "r") as f:
        ep_offset, ep_len = f["ep_offset"][:], f["ep_len"][:]
    for stride in (1, 5):
        counts = {s: len(build_window_index(ep_offset, ep_len, episodes_of_groups(split[f"{s}_groups"]), stride))
                  for s in ("train", "val")}
        print(f"stride {stride}: train {counts['train']:,} windows ({counts['train'] / 128:,.0f} steps/epoch @128), "
              f"val {counts['val']:,}")

    ds = {s: PushTWindows(args.path, s, stride=1) for s in ("train", "val")}
    out_png = Path(args.out) / "windows_grid.png"
    save_window_grid(ds, args.path, out_png)
    print(f"grid -> {out_png}")


if __name__ == "__main__":
    main()

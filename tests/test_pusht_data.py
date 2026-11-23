from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from lewm.data.pusht import (
    SPLIT_PATH,
    PushTWindows,
    build_window_index,
    episodes_of_groups,
    load_split,
    make_group_split,
)

H5 = Path("data/pusht_expert_train.h5")
needs_data = pytest.mark.skipif(not H5.exists(), reason="dataset not downloaded")


def test_group_split():
    s = make_group_split()
    train, val = set(s["train_groups"]), set(s["val_groups"])
    assert len(train) == 166 and len(val) == 19
    assert not train & val and train | val == set(range(185))
    assert make_group_split() == s
    assert load_split(SPLIT_PATH) == s  # committed file matches the seeded split


@needs_data
def test_window_index_real():
    split = load_split()
    with h5py.File(H5, "r") as f:
        off, lens, ep_idx = f["ep_offset"][:], f["ep_len"][:], f["episode_idx"][:]
    tr = build_window_index(off, lens, episodes_of_groups(split["train_groups"]))
    va = build_window_index(off, lens, episodes_of_groups(split["val_groups"]))
    assert len(tr) + len(va) == 1_981_721
    for idx in (tr, va):
        assert np.array_equal(ep_idx[idx], ep_idx[idx + 19])  # no window crosses an episode boundary
    val_eps = set(episodes_of_groups(split["val_groups"]).tolist())
    assert not set(np.unique(ep_idx[tr]).tolist()) & val_eps


@needs_data
def test_dataset_rows_match_h5():
    with h5py.File(H5, "r") as f:
        off, lens = f["ep_offset"][:], f["ep_len"][:]
        for split in ("train", "val"):
            ds = PushTWindows(H5, split)
            ep = episodes_of_groups(load_split()[f"{split}_groups"])[7]
            first, last = off[ep], off[ep] + lens[ep] - 20
            pos = {int(r): i for i, r in enumerate(ds.index)}
            picks = [pos[first], pos[last], len(ds) // 2]
            for i in picks:
                item = ds[i]
                r0 = item["r0"]
                assert item["frames"].shape == (4, 3, 224, 224) and item["frames"].dtype.is_floating_point is False
                assert item["actions"].shape == (4, 10) and item["state"].shape == (4, 7)
                for k in range(4):
                    r = r0 + 5 * k
                    assert np.array_equal(item["frames"][k].permute(1, 2, 0).numpy(), f["pixels"][r])
                    assert np.array_equal(item["actions"][k].numpy(), f["action"][r : r + 5].ravel())
                    assert np.array_equal(item["state"][k].numpy(), f["state"][r])

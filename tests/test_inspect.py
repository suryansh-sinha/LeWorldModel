"""Tests for the read-only HDF5 inspector on tiny synthetic files."""

import h5py
import numpy as np
import pytest

from lewm.data.inspect import (
    inspect_file,
    lengths_from_ends,
    lengths_from_episode_ids,
    lengths_from_offsets,
)


def test_length_helpers():
    assert lengths_from_episode_ids(np.array([0, 0, 0, 1, 1, 2])).tolist() == [3, 2, 1]
    assert lengths_from_ends(np.array([3, 5, 6])).tolist() == [3, 2, 1]
    assert lengths_from_offsets(np.array([0, 3, 5]), total_rows=6).tolist() == [3, 2, 1]


def _write_flat(path, lens, corrupt_step_idx=False):
    """Same layout as pusht_expert_train.h5: flat per-step arrays + ep_len/ep_offset/episode_idx/step_idx."""
    n = sum(lens)
    rng = np.random.default_rng(0)
    step_idx = np.concatenate([np.arange(L) for L in lens]).astype(np.int64)
    if corrupt_step_idx:
        step_idx[lens[0]] = 5  # second episode should start at step 0
    with h5py.File(path, "w") as f:
        f.create_dataset("pixels", data=rng.integers(0, 256, (n, 16, 16, 3), dtype=np.uint8), chunks=(1, 16, 16, 3), compression="gzip")
        f.create_dataset("action", data=rng.standard_normal((n, 2)).astype(np.float32))
        f.create_dataset("proprio", data=rng.standard_normal((n, 4)).astype(np.float32))
        f.create_dataset("state", data=rng.standard_normal((n, 7)).astype(np.float32))
        f.create_dataset("episode_idx", data=np.repeat(np.arange(len(lens)), lens).astype(np.int64))
        f.create_dataset("step_idx", data=step_idx)
        f.create_dataset("ep_len", data=np.array(lens, dtype=np.int32))
        f.create_dataset("ep_offset", data=np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64))


@pytest.fixture
def flat_h5(tmp_path):
    lens = [12, 7, 9]
    path = tmp_path / "tiny.h5"
    _write_flat(path, lens)
    return path, lens


def test_inspect_flat_layout(flat_h5, tmp_path):
    path, lens = flat_h5
    out = tmp_path / "out"
    rep = inspect_file(path, out)

    names = {d["name"] for d in rep["datasets"]}
    assert names == {"pixels", "action", "proprio", "state", "episode_idx", "step_idx", "ep_len", "ep_offset"}
    pix = next(d for d in rep["datasets"] if d["name"] == "pixels")
    assert pix["shape"] == [sum(lens), 16, 16, 3] and pix["dtype"] == "uint8" and pix["compression"] == "gzip"

    cands = rep["episodes"]["candidates"]
    # step_idx must NOT be mistaken for an episode id
    assert {(c["source"], c["encoding"]) for c in cands} == {
        ("ep_len", "episode lengths"),
        ("ep_offset", "episode start offsets"),
        ("episode_idx", "per-step episode id"),
    }
    for c in cands:
        assert c["num_episodes"] == 3 and c["sum_lengths"] == sum(lens)
    cons = rep["episodes"]["flat_index_consistency"]
    assert all(v for k, v in cons.items() if k.endswith(("rows", "lengths", "ep_len", "E-1", "episode")))
    assert np.load(out / "episode_lengths.npy").tolist() == lens

    img = rep["images"]["pixels"]
    assert img["encoding"] == "raw pixels" and img["height_width"] == [16, 16] and img["channels"] == 3
    assert rep["lowdim_stats"]["action"]["shape"] == [sum(lens), 2]
    assert (out / "report.json").exists() and (out / "summary.txt").exists()
    assert (out / "grid_pixels.png").exists()


def test_consistency_check_detects_bad_step_idx(tmp_path):
    path = tmp_path / "bad.h5"
    _write_flat(path, [12, 7, 9], corrupt_step_idx=True)
    cons = inspect_file(path)["episodes"]["flat_index_consistency"]
    assert cons["step_idx_is_0_to_len-1_within_episode"] is False
    assert cons["offsets_are_cumsum_of_lengths"] is True


def test_inspect_does_not_modify_file(flat_h5, tmp_path):
    path, _ = flat_h5
    before = path.read_bytes()
    inspect_file(path, tmp_path / "out")
    assert path.read_bytes() == before

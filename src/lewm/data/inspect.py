"""Read-only inspector for the PushT expert HDF5 file.

Usage:
    uv run python -m lewm.data.inspect --path data/pusht_expert_train.h5 --out outputs/inspect/

Reports the group/dataset tree (shape, dtype, chunks, filters, attrs), the episode
boundary encoding, episode-length statistics, image format, and per-dimension stats
for low-dimensional fields (actions, proprio, state). Never writes to the HDF5 file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers Blosc/LZ4/Zstd HDF5 filters if the file uses them)
import numpy as np

PAPER_NUM_EPISODES = 20_000
PAPER_MEAN_LENGTH = 196
MAX_STAT_ROWS = 10_000_000  # cap rows read per low-dim dataset when computing stats


@dataclass
class DatasetInfo:
    name: str
    shape: tuple
    dtype: str
    chunks: tuple | None
    compression: str | None
    compression_opts: object
    filters: dict
    attrs: dict
    nbytes_logical: int
    storage_bytes: int


@dataclass
class Report:
    path: str
    file_size_bytes: int
    root_attrs: dict = field(default_factory=dict)
    groups: list = field(default_factory=list)
    datasets: list = field(default_factory=list)
    episodes: dict = field(default_factory=dict)
    images: dict = field(default_factory=dict)
    lowdim_stats: dict = field(default_factory=dict)


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist() if x.size <= 64 else f"<ndarray shape={x.shape} dtype={x.dtype}>"
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, bytes):
        return x.decode(errors="replace")
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return repr(x)


def walk(f: h5py.File) -> tuple[list[str], list[DatasetInfo]]:
    """Collect every group name and dataset metadata in the file."""
    groups: list[str] = []
    datasets: list[DatasetInfo] = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group):
            groups.append(name)
        elif isinstance(obj, h5py.Dataset):
            dcpl = obj.id.get_create_plist()
            filters = {}
            for i in range(dcpl.get_nfilters()):
                fid, _flags, _vals, fname = dcpl.get_filter(i)
                filters[int(fid)] = fname.decode(errors="replace") if isinstance(fname, bytes) else str(fname)
            datasets.append(
                DatasetInfo(
                    name=name,
                    shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    chunks=obj.chunks,
                    compression=obj.compression,
                    compression_opts=_jsonable(obj.compression_opts),
                    filters=filters,
                    attrs={k: _jsonable(v) for k, v in obj.attrs.items()},
                    nbytes_logical=int(np.prod(obj.shape, dtype=np.int64)) * obj.dtype.itemsize,
                    storage_bytes=int(obj.id.get_storage_size()),
                )
            )

    f.visititems(visit)
    return groups, datasets


def lengths_from_episode_ids(ep_ids: np.ndarray) -> np.ndarray:
    """Per-step episode ids (contiguous runs) -> episode lengths in order of appearance."""
    ep_ids = np.asarray(ep_ids).reshape(-1)
    if ep_ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    change = np.flatnonzero(ep_ids[1:] != ep_ids[:-1]) + 1
    bounds = np.concatenate([[0], change, [ep_ids.size]])
    return np.diff(bounds).astype(np.int64)


def lengths_from_ends(ends: np.ndarray) -> np.ndarray:
    """Exclusive end indices (zarr-style `episode_ends`) -> lengths."""
    ends = np.asarray(ends, dtype=np.int64).reshape(-1)
    return np.diff(np.concatenate([[0], ends]))


def lengths_from_offsets(offsets: np.ndarray, total_rows: int) -> np.ndarray:
    """Start offsets of each episode -> lengths."""
    offsets = np.asarray(offsets, dtype=np.int64).reshape(-1)
    return np.diff(np.concatenate([offsets, [total_rows]]))


_EP_KEYWORDS = ("episode", "ep_", "traj")


def detect_episodes(f: h5py.File, groups: list[str], datasets: list[DatasetInfo]) -> dict:
    """Try the common boundary encodings; return every candidate found (no silent choice)."""
    candidates: list[dict] = []
    # Dominant leading dimension among datasets = total number of step rows (flat layouts).
    leading = [d.shape[0] for d in datasets if len(d.shape) >= 1]
    total_rows = int(max(set(leading), key=leading.count)) if leading else 0

    for d in datasets:
        base = d.name.split("/")[-1].lower()
        if not (base.startswith(_EP_KEYWORDS) or "len" in base):
            continue
        if len(d.shape) != 1 or not np.issubdtype(np.dtype(d.dtype), np.integer):
            continue
        arr = f[d.name][()]
        if arr.size == total_rows and total_rows > 0:
            lens = lengths_from_episode_ids(arr)
            candidates.append({"source": d.name, "encoding": "per-step episode id", "lengths": lens})
            continue
        if "len" in base and arr.sum() == total_rows:
            candidates.append({"source": d.name, "encoding": "episode lengths", "lengths": arr.astype(np.int64)})
        elif np.all(np.diff(arr) > 0) and arr.size > 0:
            if arr[-1] == total_rows:
                candidates.append({"source": d.name, "encoding": "exclusive episode ends", "lengths": lengths_from_ends(arr)})
            if arr[0] == 0:
                candidates.append(
                    {"source": d.name, "encoding": "episode start offsets", "lengths": lengths_from_offsets(arr, total_rows)}
                )

    # Per-episode group layout: many sibling groups each holding datasets.
    by_parent: dict[str, list[str]] = {}
    for g in groups:
        parent = g.rsplit("/", 1)[0] if "/" in g else ""
        by_parent.setdefault(parent, []).append(g)
    for parent, kids in by_parent.items():
        if len(kids) >= 100:
            lens = []
            for k in kids:
                ds = [f[k][n] for n in f[k] if isinstance(f[k][n], h5py.Dataset) and f[k][n].ndim >= 1]
                lens.append(ds[0].shape[0] if ds else 0)
            candidates.append({"source": parent or "/", "encoding": "one group per episode", "lengths": np.array(lens)})

    out = {"total_step_rows": total_rows, "candidates": []}
    for c in candidates:
        lens = c["lengths"]
        out["candidates"].append(
            {
                "source": c["source"],
                "encoding": c["encoding"],
                "num_episodes": int(lens.size),
                "sum_lengths": int(lens.sum()),
                "mean_length": float(lens.mean()) if lens.size else None,
                "min_length": int(lens.min()) if lens.size else None,
                "max_length": int(lens.max()) if lens.size else None,
                "median_length": float(np.median(lens)) if lens.size else None,
                "paper_num_episodes": PAPER_NUM_EPISODES,
                "paper_mean_length": PAPER_MEAN_LENGTH,
                "_lengths": lens,  # stripped before JSON dump
            }
        )
    return out


def check_flat_index_consistency(f: h5py.File) -> dict | None:
    """For the flat layout (ep_len, ep_offset, episode_idx, step_idx), verify the four agree.

    Expected: ep_offset[e] = sum(ep_len[:e]); rows ep_offset[e] : ep_offset[e]+ep_len[e] all have
    episode_idx == e (or a consistent id) and step_idx == 0..ep_len[e]-1.
    """
    keys = ("ep_len", "ep_offset", "episode_idx", "step_idx")
    if not all(k in f for k in keys):
        return None
    ep_len = f["ep_len"][()].astype(np.int64)
    ep_off = f["ep_offset"][()].astype(np.int64)
    ep_idx = f["episode_idx"][()].astype(np.int64)
    step = f["step_idx"][()].astype(np.int64)
    n_rows = ep_idx.size

    expected_off = np.concatenate([[0], np.cumsum(ep_len)[:-1]])
    expected_step = np.arange(n_rows) - np.repeat(ep_off, ep_len)  # (n_rows,)
    run_lens = lengths_from_episode_ids(ep_idx)
    first_ids = ep_idx[ep_off]
    return {
        "num_episodes": int(ep_len.size),
        "num_rows": int(n_rows),
        "sum_ep_len_equals_rows": bool(ep_len.sum() == n_rows),
        "offsets_are_cumsum_of_lengths": bool(np.array_equal(ep_off, expected_off)),
        "episode_idx_runs_equal_ep_len": bool(run_lens.size == ep_len.size and np.array_equal(run_lens, ep_len)),
        "episode_idx_at_offsets_is_0_to_E-1": bool(np.array_equal(first_ids, np.arange(ep_len.size))),
        "episode_idx_unique_values": int(np.unique(ep_idx).size),
        "step_idx_is_0_to_len-1_within_episode": bool(np.array_equal(step, expected_step)),
        "step_idx_min_max": [int(step.min()), int(step.max())],
    }


def looks_like_images(d: DatasetInfo) -> bool:
    if d.dtype == "uint8" and len(d.shape) == 4 and (d.shape[-1] in (1, 3, 4) or d.shape[1] in (1, 3, 4)):
        return True
    return d.dtype.startswith("|S") or d.dtype == "object" or "vlen" in d.dtype


def inspect_images(f: h5py.File, d: DatasetInfo, n: int = 8) -> dict:
    ds = f[d.name]
    info = {"name": d.name, "shape": d.shape, "dtype": d.dtype}
    if len(d.shape) == 4 and d.dtype == "uint8":
        idx = np.linspace(0, d.shape[0] - 1, num=min(n, d.shape[0])).astype(int)
        sample = np.stack([ds[int(i)] for i in idx])
        channels_last = d.shape[-1] in (1, 3, 4)
        info.update(
            {
                "encoding": "raw pixels",
                "layout": "(N, H, W, C)" if channels_last else "(N, C, H, W)",
                "height_width": d.shape[1:3] if channels_last else d.shape[2:4],
                "channels": d.shape[-1] if channels_last else d.shape[1],
                "sample_min": int(sample.min()),
                "sample_max": int(sample.max()),
                "sample_mean": float(sample.mean()),
                "bytes_per_frame_logical": int(np.prod(d.shape[1:])),
                "bytes_per_frame_stored": d.storage_bytes / max(d.shape[0], 1),
            }
        )
    else:
        first = ds[0]
        raw = bytes(first) if not isinstance(first, bytes) else first
        magic = raw[:8]
        fmt = "png" if magic.startswith(b"\x89PNG") else "jpeg" if magic.startswith(b"\xff\xd8") else "unknown"
        info.update({"encoding": f"encoded bytes ({fmt})", "first_item_bytes": len(raw), "magic": magic.hex()})
    return info


def lowdim_stats(f: h5py.File, d: DatasetInfo) -> dict:
    ds = f[d.name]
    rows = min(d.shape[0], MAX_STAT_ROWS)
    x = ds[:rows].astype(np.float64).reshape(rows, -1)  # (rows, feature_dim)
    return {
        "shape": d.shape,
        "dtype": d.dtype,
        "rows_used": rows,
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "any_nonfinite": bool(~np.isfinite(x).all()),
    }


def save_frame_grid(f: h5py.File, img: dict, lengths: np.ndarray | None, out_png: Path, n_eps: int = 6, n_frames: int = 8):
    """First frames of a few episodes (rows = episodes, cols = frames every 5 steps)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = f[img["name"]]
    if lengths is not None:
        n_eps = min(n_eps, len(lengths))
        all_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        eps = np.linspace(0, len(lengths) - 1, num=n_eps).astype(int)  # spread across the file
        starts, lengths = all_starts[eps], lengths[eps]
    else:
        n_eps = min(n_eps, max(ds.shape[0] // 200, 1))
        starts = np.arange(n_eps) * 200
    fig, axes = plt.subplots(n_eps, n_frames, figsize=(1.6 * n_frames, 1.6 * n_eps), squeeze=False)
    for r in range(n_eps):
        for c in range(n_frames):
            ax = axes[r, c]
            ax.axis("off")
            t = int(starts[r]) + 5 * c
            if lengths is not None and 5 * c >= lengths[r]:
                continue
            frame = ds[t]
            if img["layout"] == "(N, C, H, W)":
                frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
            ax.imshow(frame)
            if r == 0:
                ax.set_title(f"t={5 * c}", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)


def inspect_file(path: str | os.PathLike, out_dir: str | os.PathLike | None = None) -> dict:
    path = Path(path)
    with h5py.File(path, "r") as f:
        groups, datasets = walk(f)
        rep = Report(path=str(path), file_size_bytes=path.stat().st_size)
        rep.root_attrs = {k: _jsonable(v) for k, v in f.attrs.items()}
        rep.groups = groups
        rep.datasets = [d.__dict__ for d in datasets]
        rep.episodes = detect_episodes(f, groups, datasets)
        rep.episodes["flat_index_consistency"] = check_flat_index_consistency(f)

        img_ds = [d for d in datasets if looks_like_images(d)]
        rep.images = {d.name: inspect_images(f, d) for d in img_ds}
        img_names = {d.name for d in img_ds}
        for d in datasets:
            if d.name in img_names or len(d.shape) == 0 or not np.issubdtype(np.dtype(d.dtype), np.number):
                continue
            if len(d.shape) <= 3 and int(np.prod(d.shape[1:], dtype=np.int64)) <= 64:
                rep.lowdim_stats[d.name] = lowdim_stats(f, d)

        lengths = rep.episodes["candidates"][0]["_lengths"] if rep.episodes["candidates"] else None
        report = rep.__dict__
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if lengths is not None:
                np.save(out_dir / "episode_lengths.npy", lengths)
            for name, img in rep.images.items():
                if img.get("encoding") == "raw pixels":
                    save_frame_grid(f, img, lengths, out_dir / f"grid_{name.replace('/', '_')}.png")

    for c in report["episodes"]["candidates"]:
        c.pop("_lengths", None)
    report = _jsonable(report)
    if out_dir is not None:
        (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=2))
        (Path(out_dir) / "summary.txt").write_text(format_summary(report))
    return report


def format_summary(r: dict) -> str:
    gib = 2**30
    lines = [f"file: {r['path']}  ({r['file_size_bytes'] / gib:.2f} GiB on disk)", f"root attrs: {r['root_attrs']}", ""]
    lines.append("datasets:")
    for d in r["datasets"]:
        lines.append(
            f"  {d['name']}: shape={tuple(d['shape'])} dtype={d['dtype']} chunks={d['chunks']} "
            f"compression={d['compression']} filters={d['filters']} stored={d['storage_bytes'] / gib:.3f} GiB"
            + (f" attrs={d['attrs']}" if d["attrs"] else "")
        )
    lines.append(f"\ngroups ({len(r['groups'])}): {r['groups'][:20]}{' ...' if len(r['groups']) > 20 else ''}")
    lines.append(f"\nepisodes (total step rows {r['episodes']['total_step_rows']}):")
    for c in r["episodes"]["candidates"]:
        lines.append(
            f"  [{c['encoding']} @ {c['source']}] n={c['num_episodes']} (paper {c['paper_num_episodes']}), "
            f"mean len={c['mean_length']} (paper {c['paper_mean_length']}), min={c['min_length']}, "
            f"median={c['median_length']}, max={c['max_length']}, sum={c['sum_lengths']}"
        )
    if r["episodes"].get("flat_index_consistency"):
        lines.append(f"  flat index consistency: {r['episodes']['flat_index_consistency']}")
    if not r["episodes"]["candidates"]:
        lines.append("  NO episode boundary encoding detected — inspect manually")
    lines.append("\nimages:")
    for name, img in r["images"].items():
        lines.append(f"  {name}: {img}")
    lines.append("\nlow-dim stats:")
    for name, s in r["lowdim_stats"].items():
        fmt = lambda v: "[" + ", ".join(f"{x:.4g}" for x in v) + "]"  # noqa: E731
        lines.append(f"  {name} shape={tuple(s['shape'])} dtype={s['dtype']} rows_used={s['rows_used']} nonfinite={s['any_nonfinite']}")
        lines.append(f"    min={fmt(s['min'])}\n    max={fmt(s['max'])}\n    mean={fmt(s['mean'])}\n    std={fmt(s['std'])}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--path", required=True)
    p.add_argument("--out", default="outputs/inspect")
    args = p.parse_args()
    report = inspect_file(args.path, args.out)
    print(format_summary(report))


if __name__ == "__main__":
    main()

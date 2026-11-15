"""Load npz shards from pipeline/generate_dataset.py into tensors."""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_split(data_dir: str | Path, split: str):
    files = sorted(glob.glob(str(Path(data_dir) / f"{split}_*.npz")))
    if not files:
        raise FileNotFoundError(f"no {split}_*.npz in {data_dir}")
    parts = [np.load(f, allow_pickle=False) for f in files]
    keys = ["x", "vm", "va", "pf", "qf", "loading", "out_branch", "status", "stretch"]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["hour"] = np.concatenate([p["hour"] for p in parts])
    for k in ["edge_attr", "edge_index", "rated", "internal"]:
        d[k] = parts[0][k]
    keep = d["status"] == 0
    for k in keys + ["hour"]:
        d[k] = d[k][keep]
    return d


class Stats:
    """Feature normalisation fitted on the training split."""

    def __init__(self, x_mean, x_std, e_mean, e_std):
        self.x_mean, self.x_std, self.e_mean, self.e_std = x_mean, x_std, e_mean, e_std

    @classmethod
    def fit(cls, d):
        x = d["x"].reshape(-1, d["x"].shape[-1])
        e = d["edge_attr"]
        xs = x.std(0); xs[xs < 1e-6] = 1.0
        es = e.std(0); es[es < 1e-6] = 1.0
        return cls(x.mean(0), xs, e.mean(0), es)

    def state(self):
        return {k: torch.as_tensor(v) for k, v in vars(self).items()}

    @classmethod
    def from_state(cls, s):
        return cls(*(s[k].numpy() for k in ["x_mean", "x_std", "e_mean", "e_std"]))


def to_tensors(d, stats: Stats, device="cpu"):
    E = d["edge_attr"].shape[0]
    B = d["x"].shape[0]
    mask = np.ones((B, E), np.float32)
    ob = d["out_branch"]
    mask[np.arange(B)[ob >= 0], ob[ob >= 0]] = 0
    t = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=device)  # noqa: E731
    return {
        "x": t((d["x"] - stats.x_mean) / stats.x_std),
        "edge_attr": t((d["edge_attr"] - stats.e_mean) / stats.e_std),
        "edge_index": torch.as_tensor(d["edge_index"], device=device),
        "mask": t(mask),
        "vm": t(d["vm"]), "va": t(d["va"]),
        "pf": t(np.nan_to_num(d["pf"])), "qf": t(np.nan_to_num(d["qf"])),
        "loading": t(np.nan_to_num(d["loading"]) / 100.0),
        "rated": torch.as_tensor(d["rated"], device=device),
        "vset": t(d["x"][..., 3]),
    }

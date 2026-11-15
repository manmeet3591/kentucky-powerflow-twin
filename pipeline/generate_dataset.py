"""Generate AC power flow samples (N-0 and N-1) for surrogate training.

Each sample = one EIA-930 hour (LGEE/TVA load scales) x per-bus load noise
x optional stress stretch x zero or one internal branch outage, solved with
solver.KyGrid. Samples are written as npz shards so TACC array jobs can
produce them in parallel:

    python pipeline/generate_dataset.py --split train --n 2000 --shard 7

Splits (fixed, seeded):
  train / val   hours from 2023, outages drawn from the "seen" branch set
  test_seen     hours from 2024, outages from the seen branch set
  test_unseen   hours from 2024, outages only on held-out branches (10%)
Non-converged and islanding cases are recorded in the shard's `status`
array but have NaN targets.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scenarios import hourly_scales, nearest_point, align_index
from solver import KyGrid

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/dataset"
SPLIT_SEED = {"train": 1, "val": 2, "test_seen": 3, "test_unseen": 4}


def contingency_sets(g: KyGrid, seed: int = 0):
    """Internal rated branches whose single outage does not island any bus."""
    cache = ROOT / "data/grid/contingencies.json"
    if cache.exists():
        js = json.loads(cache.read_text())
        return np.array(js["seen"]), np.array(js["unseen"]), np.array(js["islanding"])
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    cand = np.where((g.branches.role.values == "internal") & g.rated)[0]
    ok, isl = [], []
    for b in cand:
        on = g.branches.status.values > 0
        on[b] = False
        A = csr_matrix((np.ones(on.sum()), (g.f[on], g.t[on])), shape=(g.n, g.n))
        ncomp, _ = connected_components(A, directed=False)
        (ok if ncomp == 1 else isl).append(int(b))
    rng = np.random.default_rng(seed)
    ok = np.array(ok)
    unseen = np.sort(rng.choice(ok, size=len(ok) // 10, replace=False))
    seen = np.setdiff1d(ok, unseen)
    cache.write_text(json.dumps({"seen": seen.tolist(), "unseen": unseen.tolist(),
                                 "islanding": isl}))
    return seen, unseen, np.array(isl)


def static_features(g: KyGrid):
    """Edge features that do not change between samples."""
    br = g.branches
    tap = np.where(br.ratio.values == 0, 1.0, br.ratio.values)
    edge = np.stack([br.r.values, br.x.values, br.b.values, tap, br.angle.values / 30.0,
                     br.rateA.values / 1000.0,
                     (br.role.values == "internal").astype(float),
                     (br.role.values == "tie").astype(float),
                     (br.role.values == "equivalent").astype(float)], axis=1).astype(np.float32)
    return edge


def node_features(g: KyGrid, pd_, qd_, pg, wx_t, wx_w):
    b = g.buses
    pgb = np.zeros(g.n)
    np.add.at(pgb, g.gbus[g.gon], pg[g.gon])
    btype = b.type.values
    x = np.stack([
        pd_ / 100, qd_ / 100, pgb / 100,
        np.where(g.pv_bus, g.vset, 0.0),
        (btype == 1).astype(float), (btype == 2).astype(float), (btype == 3).astype(float),
        g.internal.astype(float),
        b.base_kv.values / 345.0,
        g.bus_shunt_y().imag,
        (wx_t - 15.0) / 15.0, wx_w / 10.0,
    ], axis=1)
    return x.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=list(SPLIT_SEED))
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--p_n0", type=float, default=0.15, help="share of intact-topology samples")
    ap.add_argument("--p_stress", type=float, default=0.25, help="share stretched beyond observed load")
    ap.add_argument("--noise", type=float, default=0.05, help="per-bus lognormal load noise (sigma)")
    args = ap.parse_args()

    g = KyGrid.load()
    seen, unseen, _ = contingency_sets(g)
    s = align_index(hourly_scales())
    year = 2023 if args.split in ("train", "val") else 2024
    s = s[s.index.year == year]
    rng = np.random.default_rng(SPLIT_SEED[args.split] * 100_000 + args.shard)
    pool = unseen if args.split == "test_unseen" else seen
    p_n0 = 0.0 if args.split == "test_unseen" else args.p_n0

    wx_pt = nearest_point(g.buses.lat.values, g.buses.lon.values)
    edge = static_features(g)
    X, VM, VA, PF, QF, LD, OUTB, STAT, META = [], [], [], [], [], [], [], [], []
    t0 = time.time()
    for k in range(args.n):
        hour = s.index[rng.integers(len(s))]
        row = s.loc[hour]
        stretch = rng.uniform(1.0, 1.25) if rng.random() < args.p_stress else 1.0
        scale = g.load_scale_vector(row.lgee, row.tva) * stretch
        scale = scale * rng.lognormal(0, args.noise, g.n)
        ob = None if rng.random() < p_n0 else int(rng.choice(pool))
        res = g.solve(scale, out_branch=ob)
        pd_, qd_, pg = g.dispatch(scale)
        wt = np.array([row.get(f"temperature_2m_{p}", np.nan) for p in wx_pt], float)
        ww = np.array([row.get(f"wind_speed_10m_{p}", np.nan) for p in wx_pt], float)
        has_wx = np.isfinite(wt).all()
        X.append(node_features(g, pd_, qd_, pg, np.nan_to_num(wt, nan=15.0), np.nan_to_num(ww, nan=0.0)))
        ok = res.converged and res.lost_load_mw == 0
        nanb = np.full(len(edge), np.nan, np.float32)
        nann = np.full(g.n, np.nan, np.float32)
        VM.append(res.vm.astype(np.float32) if ok else nann)
        VA.append(np.deg2rad(res.va - res.va[g.ref]).astype(np.float32) if ok else nann)
        PF.append((res.p_from / 100).astype(np.float32) if ok else nanb)
        QF.append((res.q_from / 100).astype(np.float32) if ok else nanb)
        LD.append(res.loading.astype(np.float32) if ok else nanb)
        OUTB.append(-1 if ob is None else ob)
        STAT.append(0 if ok else (1 if not res.converged else 2))
        META.append((hour.isoformat(), float(stretch), bool(has_wx)))
        if (k + 1) % 50 == 0:
            print(f"{k+1}/{args.n}  {(time.time()-t0)/(k+1):.3f} s/sample", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / f"{args.split}_{args.shard:04d}.npz"
    np.savez_compressed(
        fn, x=np.stack(X), vm=np.stack(VM), va=np.stack(VA), pf=np.stack(PF), qf=np.stack(QF),
        loading=np.stack(LD), out_branch=np.array(OUTB), status=np.array(STAT),
        hour=np.array([m[0] for m in META]), stretch=np.array([m[1] for m in META]),
        has_weather=np.array([m[2] for m in META]),
        edge_attr=edge, edge_index=np.stack([g.f, g.t]).astype(np.int64),
        rated=g.rated, internal=g.internal)
    print(f"wrote {fn}  ({args.n} samples, {np.mean(np.array(STAT) == 0):.1%} usable, "
          f"{time.time()-t0:.0f} s)")


if __name__ == "__main__":
    main()

"""Full N-1 AC sweep at the dashboard's load levels -> docs/data/*.json.

For each load level we solve the intact case and every single outage of an
internal rated branch. To keep the static site small, each outage stores only
what changed versus the intact case: branch loadings that moved by more than
DL percentage points and bus voltages that moved by more than DV pu (the
browser adds these to the intact-case arrays). Summary numbers per outage
(max loading, min/max voltage, violation counts, lost load) are computed from
the full solution before thresholding.
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scenarios import hourly_scales, representative_hours
from generate_dataset import contingency_sets
from solver import KyGrid

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/data"
DL, DV = 1.5, 0.002
VLO, VHI = 0.95, 1.05          # normal voltage band (pu)
VLO_E, VHI_E = 0.90, 1.10      # post-contingency emergency band (reported too)
MON_KV = 69                    # monitored buses: Kentucky, >= 69 kV

_g = None


def _init():
    global _g
    _g = KyGrid.load()


def summarise(g, res, mon):
    L = res.loading
    vm = res.vm[mon]
    return {
        "maxL": None if np.all(np.isnan(L)) else round(float(np.nanmax(L)), 1),
        "nOver": int(np.nansum(L > 100)), "nNear": int(np.nansum((L > 90) & (L <= 100))),
        "minV": round(float(np.nanmin(vm)), 4), "maxV": round(float(np.nanmax(vm)), 4),
        "nVlow": int(np.nansum(vm < VLO)), "nVhigh": int(np.nansum(vm > VHI)),
        "nVlowE": int(np.nansum(vm < VLO_E)), "nVhighE": int(np.nansum(vm > VHI_E)),
        "lost": round(res.lost_load_mw, 1), "conv": bool(res.converged), "it": res.iterations,
    }


def run_level(level):
    g = _g
    scale = g.load_scale_vector(level["lgee"], level["tva"])
    mon = g.internal & (g.buses.base_kv.values >= MON_KV)
    t0 = time.time()
    base = g.solve(scale)
    t_base = time.time() - t0
    v0 = base.vm * np.exp(1j * np.deg2rad(base.va))
    out = {}
    times = []
    for b in level["outages"]:
        t1 = time.time()
        r = g.solve(scale, out_branch=b, v0=np.nan_to_num(v0))
        times.append(time.time() - t1)
        s = summarise(g, r, mon)
        dL = np.nan_to_num(r.loading, nan=0) - np.nan_to_num(base.loading, nan=0)
        dL[b] = 0
        ib = np.where(np.abs(dL) > DL)[0]
        dv = np.nan_to_num(r.vm, nan=0) - np.nan_to_num(base.vm, nan=0)
        iv = np.where(np.abs(dv) > DV)[0]
        s["dL"] = [[int(i), round(float(np.nan_to_num(r.loading[i], nan=0)), 1)] for i in ib]
        s["dV"] = [[int(i), round(float(r.vm[i]), 4) if np.isfinite(r.vm[i]) else None] for i in iv]
        out[str(b)] = s
    pack = {
        "key": level["key"],
        "base": {"loading": [None if np.isnan(x) else round(float(x), 1) for x in base.loading],
                 "vm": [round(float(x), 4) for x in base.vm],
                 "pf": [round(float(x), 1) for x in base.p_from],
                 **summarise(g, base, mon)},
        "outages": out,
        "timing": {"base_s": round(t_base, 3), "outage_mean_s": round(float(np.mean(times)), 4),
                   "n": len(times)},
    }
    (DOCS / f"level_{level['key']}.json").write_text(json.dumps(pack, separators=(",", ":")))
    worst = sorted(out.items(), key=lambda kv: -(kv[1]["maxL"] or 0))[:3]
    print(level["key"], "done", f"{time.time()-t0:.0f}s", "base maxL", pack["base"]["maxL"],
          "worst", [(k, v["maxL"], v["nOver"]) for k, v in worst], flush=True)
    return level["key"], pack["timing"]


def display_coords(g):
    """Buses whose gazetteer match lands far from every neighbour (a same-named town in
    another state) are moved to the mean of their neighbours, for display only."""
    lat, lon = g.buses.lat.values.copy(), g.buses.lon.values.copy()
    real = g.branches.role.values != "equivalent"
    nb = [[] for _ in range(g.n)]
    for a, b in zip(g.f[real], g.t[real]):
        nb[a].append(b); nb[b].append(a)
    moved = 0
    for _ in range(3):
        for i in range(g.n):
            if not nb[i]:
                continue
            ml, mo = np.mean(lat[nb[i]]), np.mean(lon[nb[i]])
            if np.hypot(lat[i] - ml, (lon[i] - mo) * 0.8) > 1.0:
                lat[i], lon[i] = ml + 0.03, mo + 0.03
                moved += 1
    print("display coords: moved", moved, "buses")
    return lat, lon


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    g = KyGrid.load()
    seen, unseen, isl = contingency_sets(g)
    if "--grid-only" in sys.argv:
        write_grid(g, seen, unseen, isl)
        return
    outages = sorted(np.r_[seen, unseen, isl].astype(int).tolist())
    s = hourly_scales()
    levels = representative_hours(s)
    for lv in levels:
        lv["outages"] = outages
    n_proc = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6
    with Pool(n_proc, initializer=_init) as pool:
        timing = dict(pool.map(run_level, levels))
    meta = [{k: v for k, v in lv.items() if k != "outages"} | {"timing": timing[lv["key"]]}
            for lv in levels]
    (DOCS / "levels.json").write_text(json.dumps(meta, indent=1))
    write_grid(g, seen, unseen, isl)


def write_grid(g, seen, unseen, isl):
    """Static grid description for the map."""
    b, br = g.buses, g.branches
    lat, lon = display_coords(g)
    kv = b.base_kv.values
    name = b.name.values
    grid = {
        "buses": [{"i": i, "id": int(b.bus_id[i]), "name": name[i], "kv": float(kv[i]),
                   "role": b.role[i], "lat": round(float(lat[i]), 4), "lon": round(float(lon[i]), 4),
                   "coord": b.coord_source[i]} for i in range(len(b))],
        "branches": [{"i": int(k), "f": int(g.f[k]), "t": int(g.t[k]),
                      "kv": float(max(kv[g.f[k]], kv[g.t[k]])),
                      "xfmr": bool(br.ratio[k] != 0 or kv[g.f[k]] != kv[g.t[k]]),
                      "rate": float(br.rateA[k]), "role": br.role[k],
                      "name": f"{name[g.f[k]]} – {name[g.t[k]]}"} for k in range(len(br))],
        "contingencies": {"seen": seen.tolist(), "unseen": unseen.tolist(), "islanding": isl.tolist()},
        "limits": {"VLO": VLO, "VHI": VHI, "VLO_E": VLO_E, "VHI_E": VHI_E, "DL": DL, "DV": DV,
                   "MON_KV": MON_KV},
        "summary": json.loads((ROOT / "data/grid/summary.json").read_text()),
    }
    (DOCS / "grid.json").write_text(json.dumps(grid, separators=(",", ":")))
    print("wrote", DOCS)


if __name__ == "__main__":
    main()

"""Cut a Kentucky-region network out of the ACTIVSg70k synthetic Eastern US case.

Area 27 of ACTIVSg70k is the synthetic grid on the Kentucky footprint (85% of its
bus names are Kentucky place names; see README). Everything outside area 27 is
replaced by an extended-Ward style equivalent:

  * the first ring of external buses (directly tied to area 27) is kept as
    "boundary" buses;
  * every other external bus is eliminated by Kron reduction of the full
    70k-bus admittance matrix, which leaves equivalent branches and shunts
    between boundary buses;
  * each boundary bus gets a PV "equivalent generator" whose P and V reproduce
    the stored ACTIVSg70k solution exactly.

Outputs (data/grid/):
  ky_case.m            MATPOWER case of the reduced network (for gridfm-datakit / PowerModels)
  ky_net.json          pandapower network
  buses.csv, branches.csv, gens.csv   tables with roles and approximate coordinates
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import splu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
from matpower_io import read_case, write_case  # noqa: E402

KY_AREA = 27
EQ_MIN_Y = 0.2      # keep equivalent branches with |y| > 0.2 pu (x_eq < 5 pu)
GAZ = ROOT / "data/raw/gazetteer"
OUT = ROOT / "data/grid"

# state that dominates each neighbouring area (from the same name-match test)
AREA_STATE = {26: "TN", 28: "WV", 29: "OH", 30: "OH", 33: "IN", 34: "IL", 35: "IL",
              40: "MO", 41: "MO", 15: "VA", 14: "VA"}


def ybus(bus, branch, idx):
    n = len(bus)
    f = np.array([idx[int(x)] for x in branch[:, 0]])
    t = np.array([idx[int(x)] for x in branch[:, 1]])
    ys = 1 / (branch[:, 2] + 1j * branch[:, 3])
    bc = branch[:, 4]
    tap = np.where(branch[:, 8] == 0, 1, branch[:, 8]) * np.exp(1j * np.deg2rad(branch[:, 9]))
    ytt = ys + 1j * bc / 2
    yff = ytt / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    Y = csr_matrix((np.r_[yff, yft, ytf, ytt], (np.r_[f, f, t, t], np.r_[f, t, f, t])), shape=(n, n))
    Y = Y + csr_matrix(((bus[:, 4] + 1j * bus[:, 5]) / 100, (np.arange(n), np.arange(n))), shape=(n, n))
    return Y.tocsr()


def gazetteer():
    places = defaultdict(dict)
    for fn in ["2023_Gaz_place_national.txt", "2023_Gaz_cousubs_national.txt"]:
        with open(GAZ / fn, encoding="latin-1") as fh:
            for row in csv.reader(fh, delimiter="\t"):
                if row[0] == "USPS":
                    continue
                nm = re.sub(r"\s+(city|town|village|CDP|borough|township|municipality|.*\(balance\))$",
                            "", row[3].strip(), flags=re.I).upper()
                places[nm].setdefault(row[0], (float(row[-2]), float(row[-1])))
    return places


def city_of(name: str) -> str:
    return re.sub(r"\s+\d+\s+\d+$", "", name).strip()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = read_case(ROOT / "data/raw/case_ACTIVSg70k.m")
    bus, gen, br = c["bus"], c["gen"], c["branch"]
    names = c["bus_name"]
    ids = bus[:, 0].astype(int)
    idx = {b: i for i, b in enumerate(ids)}
    n = len(ids)

    V = bus[:, 7] * np.exp(1j * np.deg2rad(bus[:, 8]))
    Y = ybus(bus, br, idx)
    on = gen[:, 7] > 0
    Sg = np.zeros(n, complex)
    np.add.at(Sg, [idx[int(x)] for x in gen[on, 0]], (gen[on, 1] + 1j * gen[on, 2]) / 100)
    Sd = (bus[:, 2] + 1j * bus[:, 3]) / 100
    mis = V * np.conj(Y @ V) - (Sg - Sd)
    print(f"stored 70k solution: max P mismatch {abs(mis.real).max()*100:.2f} MW")

    internal = bus[:, 6] == KY_AREA
    fi = np.array([idx[int(x)] for x in br[:, 0]])
    ti = np.array([idx[int(x)] for x in br[:, 1]])
    tie = internal[fi] ^ internal[ti]
    bnd = np.zeros(n, bool)
    bnd[fi[tie & ~internal[fi]]] = True
    bnd[ti[tie & ~internal[ti]]] = True
    far = ~internal & ~bnd
    I, B, F = np.where(internal)[0], np.where(bnd)[0], np.where(far)[0]
    print(f"internal {len(I)}  boundary {len(B)}  eliminated {len(F)}  tie branches {tie.sum()}")

    # Kron-reduce the far external network onto the boundary buses
    Ycsc = Y.tocsc()
    YBB = Ycsc[B][:, B].toarray()
    YBF = Ycsc[B][:, F]
    YFB = Ycsc[F][:, B].toarray()
    lu = splu(csc_matrix(Ycsc[F][:, F]))
    X = lu.solve(YFB)
    Yeq = YBB - (YBF @ X)
    Yeq = 0.5 * (Yeq + Yeq.T)  # tiny asymmetry from phase shifters

    # reduced bus set (keep original ids and names)
    keep = np.r_[I, B]
    kb = bus[keep, :13].copy()
    kb[len(I):, 2:4] = 0          # boundary loads are folded into the equivalent injection
    kb[len(I):, 4:6] = 0
    kb[:, 1] = np.where(kb[:, 1] == 3, 2, kb[:, 1])

    # branches: everything inside area 27 plus tie lines (kept as-is)
    in_or_tie = (internal[fi] & internal[ti]) | tie
    kbr = br[in_or_tie, :13].copy()
    roles = ["internal" if (internal[a] and internal[b]) else "tie" for a, b in zip(fi[in_or_tie], ti[in_or_tie])]

    # equivalent branches + shunts between boundary buses
    eq_rows, shunt = [], np.zeros(len(B), complex)
    nB = len(B)
    for a in range(nB):
        for b in range(a + 1, nB):
            y = -Yeq[a, b]
            if abs(y) > EQ_MIN_Y:
                z = 1 / y
                # Kron reduction gives some negative resistances; clip so equivalents never
                # generate power (the injections below re-match the base solution anyway)
                z = complex(max(z.real, 0.0), z.imag)
                eq_rows.append([ids[B[a]], ids[B[b]], z.real, z.imag, 0, 0, 0, 0, 0, 0, 1, -360, 360])
    eq_rows = np.array(eq_rows)
    kept_y = np.zeros((nB, nB), complex)
    pos = {ids[B[k]]: k for k in range(nB)}
    for r in eq_rows:
        a, b = pos[int(r[0])], pos[int(r[1])]
        y = 1 / (r[2] + 1j * r[3])
        kept_y[a, b] = kept_y[b, a] = y
    # The Kron row sums imply very large equivalent shunts (hundreds of GW of P at 1 pu
    # when phase shifters and off-nominal taps are folded in), which would make the
    # boundary hypersensitive to voltage. We drop them: boundary buses are PV, and the
    # injections computed below absorb what the shunts would have carried at the base point.
    shunt = np.zeros(nB, complex)
    kb[len(I):, 4] = shunt.real * 100
    kb[len(I):, 5] = shunt.imag * 100
    kbr = np.vstack([kbr, eq_rows])
    roles += ["equivalent"] * len(eq_rows)

    # internal generators as in the case; boundary equivalent generators reproduce the solution
    kidx = {b: i for i, b in enumerate(kb[:, 0].astype(int))}
    gmask = internal[[idx[int(x)] for x in gen[:, 0]]]
    kgen = gen[gmask].copy()
    gfuel = [c["genfuel"][i] for i in np.where(gmask)[0]]
    # Some units sat on Q limits in the stored solution, so their bus voltage is below
    # the case's Vg setpoint. Use the solved voltage as setpoint so a PV solve (no Q
    # limits) reproduces the base case.
    kgen[:, 5] = np.abs(V[[idx[int(x)] for x in kgen[:, 0]]])
    Yr = ybus(kb, kbr, kidx)
    Vk = V[keep]
    Sinj = Vk * np.conj(Yr @ Vk)
    eq_gen = []
    for k in range(len(I), len(keep)):
        s = Sinj[k] * 100
        eq_gen.append([kb[k, 0], s.real, s.imag, 9999, -9999, abs(Vk[k]), 100, 1, 9999, -9999])
    eq_gen = np.array(eq_gen)
    kb[len(I):, 1] = 2
    # slack: the boundary bus with the highest kV and largest |P| injection
    cand = np.lexsort((-np.abs(eq_gen[:, 1]), -kb[len(I):, 9]))[0]
    kb[len(I) + cand, 1] = 3
    allgen = np.vstack([kgen[:, :10], eq_gen])
    fuel = gfuel + ["equivalent"] * len(eq_gen)

    # internal gen buses that lost all units -> PQ
    gb = set(allgen[allgen[:, 7] > 0, 0].astype(int))
    for k in range(len(kb)):
        if kb[k, 1] == 2 and int(kb[k, 0]) not in gb:
            kb[k, 1] = 1

    # verify the reduced network reproduces the 70k solution
    Sgk = np.zeros(len(kb), complex)
    on = allgen[:, 7] > 0
    np.add.at(Sgk, [kidx[int(x)] for x in allgen[on, 0]], (allgen[on, 1] + 1j * allgen[on, 2]) / 100)
    misk = Vk * np.conj(Yr @ Vk) - (Sgk - (kb[:, 2] + 1j * kb[:, 3]) / 100)
    print(f"reduced case at stored solution: max P mismatch {abs(misk.real).max()*100:.2f} MW, "
          f"max Q mismatch (PQ buses) {abs(misk.imag[kb[:, 1] == 1]).max()*100:.2f} Mvar")

    case = {"baseMVA": 100.0, "bus": kb, "gen": allgen, "branch": kbr,
            "bus_name": [names[i] for i in keep], "genfuel": fuel}
    # linear cost so OPF tools (datakit opf mode) accept the case; equivalents are expensive
    gc = np.zeros((len(allgen), 7))
    gc[:, 0] = 2; gc[:, 3] = 3
    origcost = c["gencost"][gmask]
    gc[:len(kgen), 4:7] = origcost[:, 4:7]
    gc[len(kgen):, 5] = 60.0
    case["gencost"] = gc
    write_case(case, OUT / "ky_case.m", "ky_case",
               header=" Kentucky-region cut of ACTIVSg70k (area 27) with Ward-equivalent boundary.\n"
                      " Synthetic; not the actual Kentucky grid. Generated by pipeline/extract_kentucky.py")

    # coordinates from Census gazetteer place names (approximate: town centroid)
    places = gazetteer()
    lat, lon, how = [], [], []
    for k, i in enumerate(keep):
        st = "KY" if internal[i] else AREA_STATE.get(int(bus[i, 6]))
        p = places.get(city_of(names[i]), {})
        if st in p:
            la, lo = p[st]; h = "gazetteer"
        elif len(p) == 1 and not internal[i]:
            la, lo = next(iter(p.values())); h = "gazetteer-other-state"
        else:
            la, lo, h = np.nan, np.nan, "none"
        lat.append(la); lon.append(lo); how.append(h)
    lat, lon = np.array(lat), np.array(lon)
    # Kentucky buses must land inside the state's bounding box
    KY_BOX = (36.45, 39.15, -89.6, -81.9)
    for k, i in enumerate(keep):
        if internal[i] and not (KY_BOX[0] <= lat[k] <= KY_BOX[1] and KY_BOX[2] <= lon[k] <= KY_BOX[3]):
            lat[k] = lon[k] = np.nan; how[k] = "none"
    # unmatched buses: average of matched neighbours (repeat to propagate)
    nb = defaultdict(set)
    for r in kbr[np.array(roles) != "equivalent"]:
        a, b = kidx[int(r[0])], kidx[int(r[1])]
        nb[a].add(b); nb[b].add(a)
    for _ in range(10):
        for k in np.where(np.isnan(lat))[0]:
            ok = [j for j in nb[k] if not np.isnan(lat[j])]
            if ok:
                lat[k] = np.mean(lat[ok]); lon[k] = np.mean(lon[ok]); how[k] = "neighbour-mean"
    # same-substation buses share a point; jitter slightly for display
    rng = np.random.default_rng(0)
    jit = rng.normal(0, 0.01, (len(keep), 2))

    bdf = pd.DataFrame({
        "bus_id": kb[:, 0].astype(int), "name": case["bus_name"],
        "role": ["internal"] * len(I) + ["boundary"] * len(B),
        "type": kb[:, 1].astype(int), "base_kv": kb[:, 9], "pd_mw": kb[:, 2], "qd_mvar": kb[:, 3],
        "vm_base": abs(Vk), "va_base_deg": np.rad2deg(np.angle(Vk)),
        "lat": lat + jit[:, 0], "lon": lon + jit[:, 1], "coord_source": how,
        "orig_area": bus[keep, 6].astype(int),
    })
    bdf.to_csv(OUT / "buses.csv", index=False)
    brdf = pd.DataFrame(kbr[:, :11], columns=["fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC",
                                               "ratio", "angle", "status"])
    brdf.insert(0, "branch_id", np.arange(len(brdf)))
    brdf["role"] = roles
    brdf.to_csv(OUT / "branches.csv", index=False)
    gdf = pd.DataFrame(allgen[:, :10], columns=["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin"])
    gdf["fuel"] = fuel
    gdf.to_csv(OUT / "gens.csv", index=False)
    summary = {
        "source": "ACTIVSg70k (TAMU, MATPOWER data/case_ACTIVSg70k.m), area 27",
        "internal_buses": int(len(I)), "boundary_buses": int(len(B)), "eliminated_buses": int(len(F)),
        "internal_branches": int(roles.count("internal")), "tie_branches": int(roles.count("tie")),
        "equivalent_branches": int(roles.count("equivalent")),
        "internal_load_mw": float(kb[:len(I), 2].sum()),
        "internal_gen_mw": float(kgen[kgen[:, 7] > 0, 1].sum()),
        "internal_gen_pmax_mw": float(kgen[kgen[:, 7] > 0, 8].sum()),
        "net_import_mw": float(eq_gen[:, 1].sum()),
        "coord_sources": dict(Counter(how)),
        "max_p_mismatch_mw_at_stored_solution": float(abs(misk.real).max() * 100),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

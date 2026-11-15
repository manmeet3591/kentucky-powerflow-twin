"""Cross-check gridfm-datakit (PowerModels.jl) solutions against solver.KyGrid's network model.

Reads a datakit output dir, plugs each PowerModels solution into our admittance
model and reports active-power mismatch and from-end flow differences.
datakit numbers buses by sorted bus id; branches keep the ky_case.m order.

    python pipeline/check_datakit.py data/datakit_tiny/ky_case/raw
"""
import sys

import numpy as np
import pandas as pd
from scipy.sparse import diags

from solver import KyGrid

raw = sys.argv[1] if len(sys.argv) > 1 else "data/datakit_tiny/ky_case/raw"
g = KyGrid.load()
b = pd.read_parquet(f"{raw}/bus_data.parquet")
br = pd.read_parquet(f"{raw}/branch_data.parquet")
pos = np.empty(g.n, int)
pos[np.argsort(g.buses.bus_id.values)] = np.arange(g.n)
for s in sorted(b.scenario.unique()):
    bs, bb = b[b.scenario == s].sort_values("bus"), br[br.scenario == s].sort_values("idx")
    V = (bs.Vm.values * np.exp(1j * np.deg2rad(bs.Va.values)))[pos]
    on = bb.br_status.values > 0
    Y = g.ybus(on, 1.0) - diags(g.bus_shunt_y()) + diags(((bs.GS.values + 1j * bs.BS.values) / 100)[pos])
    p_mis = (V * np.conj(Y @ V) * 100).real - (bs.Pg - bs.Pd).values[pos]
    Sf = V[g.f] * np.conj(g.yff * V[g.f] + g.yft * V[g.t]) * 100
    print(f"scenario {s}: outages {np.where(~on)[0].tolist()}  load {bs.Pd.sum():.0f} MW  "
          f"max |P mismatch| {np.abs(p_mis).max():.4f} MW  max |pf diff| {np.abs(Sf.real[on] - bb.pf.values[on]).max():.4f} MW")

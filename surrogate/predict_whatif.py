"""Run the trained surrogate on every dashboard what-if case.

Writes docs/data/gnn_level_<key>.json in the same delta format as the solver
files, so the dashboard can show solver vs surrogate side by side. The
dashboard only offers the surrogate once these files exist.

    python surrogate/predict_whatif.py --ckpt runs/gnn/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "surrogate"))
from evaluate import load_model, set_manifest  # noqa: E402
from generate_dataset import node_features, static_features  # noqa: E402
from precompute_whatif import DL, DV, MON_KV, VHI, VHI_E, VLO, VLO_E  # noqa: E402
from scenarios import nearest_point  # noqa: E402
from solver import KyGrid  # noqa: E402

DOCS = ROOT / "docs/data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/gnn/best.pt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, st = load_model(args.ckpt, dev)
    g = KyGrid.load()
    levels = json.loads((DOCS / "levels.json").read_text())
    grid = json.loads((DOCS / "grid.json").read_text())
    c = grid["contingencies"]
    outages = sorted(c["seen"] + c["unseen"] + c["islanding"])
    edge = torch.tensor((static_features(g) - st.e_mean) / st.e_std, dtype=torch.float32, device=dev)
    ei = torch.as_tensor(np.stack([g.f, g.t]), device=dev)
    pts = nearest_point(g.buses.lat.values, g.buses.lon.values)
    mon = g.internal & (g.buses.base_kv.values >= MON_KV)
    for lv in levels:
        scale = g.load_scale_vector(lv["lgee"], lv["tva"])
        pd_, qd_, pg = g.dispatch(scale)
        w = lv.get("weather", {})
        wt = np.array([w.get(f"temperature_2m_{p}", 15.0) for p in pts])
        ww = np.array([w.get(f"wind_speed_10m_{p}", 0.0) for p in pts])
        x = (node_features(g, pd_, qd_, pg, wt, ww) - st.x_mean) / st.x_std
        cases = [None] + outages
        X = torch.tensor(x, dtype=torch.float32, device=dev).unsqueeze(0)
        res_vm, res_ld = [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(cases), args.batch):
                chunk = cases[i:i + args.batch]
                mask = torch.ones(len(chunk), len(g.f), device=dev)
                for j, b in enumerate(chunk):
                    if b is not None:
                        mask[j, b] = 0
                node, e = model(X.expand(len(chunk), -1, -1), edge, ei, mask)
                res_vm.append((1 + node[..., 0] * 0.05).cpu().numpy())
                res_ld.append((e[..., 2] * 100).cpu().numpy())
        ms = 1000 * (time.time() - t0) / len(cases)
        vm, ld = np.concatenate(res_vm), np.concatenate(res_ld)
        ld = np.where(g.rated, ld, np.nan)
        base_vm, base_ld = vm[0], ld[0]

        def summ(v, L):
            vv = v[mon]
            return {"maxL": round(float(np.nanmax(L)), 1), "nOver": int(np.nansum(L > 100)),
                    "nNear": int(np.nansum((L > 90) & (L <= 100))),
                    "minV": round(float(vv.min()), 4), "maxV": round(float(vv.max()), 4),
                    "nVlow": int((vv < VLO).sum()), "nVhigh": int((vv > VHI).sum()),
                    "nVlowE": int((vv < VLO_E).sum()), "nVhighE": int((vv > VHI_E).sum()),
                    "lost": None, "conv": True}
        out = {}
        for k, b in enumerate(outages, start=1):
            L = ld[k].copy(); L[b] = np.nan
            s = summ(vm[k], L)
            dL = np.nan_to_num(L) - np.nan_to_num(base_ld)
            dL[b] = 0
            s["dL"] = [[int(i), round(float(np.nan_to_num(L[i])), 1)] for i in np.where(np.abs(dL) > DL)[0]]
            s["dV"] = [[int(i), round(float(vm[k][i]), 4)] for i in np.where(np.abs(vm[k] - base_vm) > DV)[0]]
            out[str(b)] = s
        pack = {"key": lv["key"], "source": "gnn", "ckpt": Path(args.ckpt).name,
                "base": {"loading": [None if np.isnan(v) else round(float(v), 1) for v in base_ld],
                         "vm": [round(float(v), 4) for v in base_vm], **summ(base_vm, base_ld)},
                "outages": out, "timing": {"ms_per_case": round(ms, 2), "device": str(dev)}}
        (DOCS / f"gnn_level_{lv['key']}.json").write_text(json.dumps(pack, separators=(",", ":")))
        print(lv["key"], f"{ms:.2f} ms/case on {dev}")
    set_manifest(gnn=True)


if __name__ == "__main__":
    main()

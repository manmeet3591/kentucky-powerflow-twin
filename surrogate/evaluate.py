"""Evaluate the surrogate against the AC solver on held-out scenarios.

    python surrogate/evaluate.py --ckpt runs/gnn/best.pt --data data/dataset

Test splits: test_seen (2024 hours, outages on lines also outaged in training)
and test_unseen (2024 hours, outages only on the 10% of lines never outaged in
training). Reports, per split, for the GNN and for a DC power flow baseline:
  Vm MAE (pu), Va MAE (deg), loading MAE (percentage points, rated branches),
  overload (>100%) precision/recall, voltage-violation recall (<0.95 or >1.05),
  and inference time per scenario (batched and single).
Writes <ckpt dir>/eval.json and docs/data/surrogate_eval.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Stats, load_split, to_tensors  # noqa: E402
from model import PFSurrogate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def set_manifest(**kw):
    """docs/data/manifest.json tells the dashboard which surrogate outputs exist."""
    fn = ROOT / "docs/data/manifest.json"
    m = json.loads(fn.read_text()) if fn.exists() else {}
    m.update(kw)
    fn.write_text(json.dumps(m))


def load_model(ckpt, device):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck["config"]
    m = PFSurrogate(c["n_node_feat"], c["n_edge_feat"], c["hidden"], c["layers"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, Stats.from_state(ck["stats"])


def predict(model, T, batch, device):
    vm, va, ld = [], [], []
    with torch.no_grad():
        for i in range(0, len(T["x"]), batch):
            sl = slice(i, i + batch)
            node, edge = model(T["x"][sl], T["edge_attr"], T["edge_index"], T["mask"][sl])
            vm.append((1 + node[..., 0] * 0.05).cpu()); va.append(node[..., 1].cpu())
            ld.append((edge[..., 2] * 100).cpu())
    return torch.cat(vm).numpy(), torch.cat(va).numpy(), torch.cat(ld).numpy()


def dc_baseline(d):
    """DC power flow loading (|P|/rateA) from the same injections the GNN sees."""
    br = pd.read_csv(ROOT / "data/grid/branches.csv")
    f, t = d["edge_index"]
    n = d["x"].shape[1]
    tap = np.where(br.ratio.values == 0, 1, br.ratio.values)
    bser = 1 / (br.x.values * tap)
    shift = np.deg2rad(br.angle.values)
    ref = int(np.where(d["x"][0, :, 6] > 0.5)[0][0])
    rate = br.rateA.values
    out = np.full(d["loading"].shape, np.nan, np.float32)
    keep = np.setdiff1d(np.arange(n), [ref])
    for s in range(len(d["x"])):
        on = np.ones(len(f), bool)
        if d["out_branch"][s] >= 0:
            on[d["out_branch"][s]] = False
        bb = bser * on
        B = csr_matrix((np.r_[bb, -bb, -bb, bb], (np.r_[f, f, t, t], np.r_[f, t, f, t])), shape=(n, n)).tocsc()
        P = (d["x"][s, :, 2] - d["x"][s, :, 0])        # pu (features are MW/100)
        Pshift = np.zeros(n)
        np.add.at(Pshift, f, -bb * shift); np.add.at(Pshift, t, bb * shift)
        th = np.zeros(n)
        th[keep] = splu(B[keep][:, keep]).solve((P - Pshift)[keep])
        pf = bb * (th[f] - th[t] - shift) * 100
        out[s] = np.where(d["rated"], 100 * np.abs(pf) / np.where(rate > 0, rate, 1), np.nan)
    return out


def metrics(d, vm, va, ld):
    rated = d["rated"][None, :] & ~np.isnan(d["loading"])
    if (d["out_branch"] >= 0).any():
        rated = rated.copy()
        s = np.where(d["out_branch"] >= 0)[0]
        rated[s, d["out_branch"][s]] = False
    L, Lp = d["loading"][rated], ld[rated]
    over, over_p = L > 100, Lp > 100
    tp = (over & over_p).sum()
    # monitored buses: Kentucky, >= 69 kV (same rule as the dashboard)
    kv = d["x"][0, :, 8] * 345.0
    mon = (d["internal"] & (kv >= 68.5))[None, :].repeat(len(d["vm"]), 0)
    viol = ((d["vm"] < 0.95) | (d["vm"] > 1.05)) & mon
    m = {
        "loading_mae_pp": float(np.abs(Lp - L).mean()),
        "loading_p95_err_pp": float(np.percentile(np.abs(Lp - L), 95)),
        "overloads_true": int(over.sum()), "overloads_pred": int(over_p.sum()),
        "overload_recall": float(tp / max(over.sum(), 1)),
        "overload_precision": float(tp / max(over_p.sum(), 1)),
    }
    if vm is not None:
        m["vm_mae_pu"] = float(np.abs(vm - d["vm"])[mon].mean())
        m["va_mae_deg"] = float(np.rad2deg(np.abs(va - d["va"])).mean())
        vp = ((vm < 0.95) | (vm > 1.05)) & mon
        m["vviol_true"] = int(viol.sum())
        m["vviol_recall"] = float((viol & vp).sum() / max(viol.sum(), 1))
        m["vviol_precision"] = float((viol & vp).sum() / max(vp.sum(), 1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/gnn/best.pt"))
    ap.add_argument("--data", default=str(ROOT / "data/dataset"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no_docs", action="store_true")
    args = ap.parse_args()
    dev = torch.device(args.device)
    model, stats = load_model(args.ckpt, dev)
    report = {"ckpt": args.ckpt, "device": str(dev)}
    for split in ["test_seen", "test_unseen"]:
        d = load_split(args.data, split)
        T = to_tensors(d, stats, dev)
        predict(model, {k: (v[:2] if k in ("x", "mask") else v) for k, v in T.items()}, 2, dev)  # warm-up
        t0 = time.time(); vm, va, ld = predict(model, T, args.batch, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_batch = (time.time() - t0) / len(d["x"])
        t0 = time.time(); predict(model, {k: (v[:20] if k in ("x", "mask") else v) for k, v in T.items()}, 1, dev)
        t_single = (time.time() - t0) / min(20, len(d["x"]))
        report[split] = {
            "n": int(len(d["x"])),
            "gnn": metrics(d, vm, va, ld) | {"ms_per_scenario_batched": 1000 * t_batch,
                                             "ms_per_scenario_single": 1000 * t_single},
            "dc_pf": metrics(d, None, None, dc_baseline(d)),
        }
        print(split, json.dumps(report[split], indent=1), flush=True)
    out = Path(args.ckpt).parent / "eval.json"
    out.write_text(json.dumps(report, indent=1))
    if not args.no_docs:
        (ROOT / "docs/data/surrogate_eval.json").write_text(json.dumps(report, indent=1))
        set_manifest(eval=True)


if __name__ == "__main__":
    main()

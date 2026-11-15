"""Train the GNN power flow surrogate.

    python surrogate/train.py --data data/dataset --out runs/gnn --epochs 200   # TACC GPU
    python surrogate/train.py --data data/dataset --out runs/sanity --epochs 2 --hidden 32 --layers 3  # CPU check

Loss = MSE on Vm (x100, so 0.01 pu ~ 1), Va (rad x10), P/Q from-end flows (pu),
loading/100 on rated in-service branches, plus an optional physics term that
compares flows computed from the predicted voltages with the solver flows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Stats, load_split, to_tensors  # noqa: E402
from model import PFSurrogate, physics_flows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def branch_admittances(device):
    import pandas as pd
    br = pd.read_csv(ROOT / "data/grid/branches.csv")
    ys = 1 / (br.r.values + 1j * br.x.values)
    tap = np.where(br.ratio.values == 0, 1, br.ratio.values) * np.exp(1j * np.deg2rad(br.angle.values))
    ytt = ys + 1j * br.b.values / 2
    y = {"yff": ytt / (tap * np.conj(tap)), "yft": -ys / np.conj(tap), "ytf": -ys / tap, "ytt": ytt}
    return {k: (torch.tensor(v.real, dtype=torch.float32, device=device),
                torch.tensor(v.imag, dtype=torch.float32, device=device)) for k, v in y.items()}


def losses(model, T, idx, adm, w_phys):
    x, mask = T["x"][idx], T["mask"][idx]
    node, edge = model(x, T["edge_attr"], T["edge_index"], mask)
    vm = 1.0 + node[..., 0] * 0.05
    va = node[..., 1]
    l_vm = ((vm - T["vm"][idx]) * 100).pow(2).mean()
    l_va = ((va - T["va"][idx]) * 10).pow(2).mean()
    rated = T["rated"].unsqueeze(0) & (mask > 0)
    l_pf = (edge[..., 0] - T["pf"][idx]).pow(2).mean() + (edge[..., 1] - T["qf"][idx]).pow(2).mean()
    l_ld = ((edge[..., 2] - T["loading"][idx]).pow(2) * rated).sum() / rated.sum().clamp(min=1)
    loss = l_vm + l_va + l_pf + 10 * l_ld
    out = {"vm": l_vm.item(), "va": l_va.item(), "pf": l_pf.item(), "ld": l_ld.item()}
    if w_phys > 0:
        src, dst = T["edge_index"]
        Sf, _ = physics_flows(vm, va, adm, src, dst, mask)
        l_ph = (Sf.real - T["pf"][idx]).pow(2).mean() + (Sf.imag - T["qf"][idx]).pow(2).mean()
        loss = loss + w_phys * l_ph
        out["phys"] = l_ph.item()
    return loss, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/dataset"))
    ap.add_argument("--out", default=str(ROOT / "runs/gnn"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--w_phys", type=float, default=0.1)
    ap.add_argument("--max_train", type=int, default=0, help="cap training samples (0 = all)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    dev = torch.device(args.device)

    tr, va = load_split(args.data, "train"), load_split(args.data, "val")
    if args.max_train:
        for k in ["x", "vm", "va", "pf", "qf", "loading", "out_branch", "status", "stretch", "hour"]:
            tr[k] = tr[k][:args.max_train]
    stats = Stats.fit(tr)
    T, V = to_tensors(tr, stats, dev), to_tensors(va, stats, dev)
    adm = branch_admittances(dev)
    model = PFSurrogate(T["x"].shape[-1], T["edge_attr"].shape[-1], args.hidden, args.layers).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"train {len(tr['x'])}  val {len(va['x'])}  params {n_par:,}  device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=max(1, len(tr["x"]) // args.batch))
    best, log = 1e9, []
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(tr["x"]), device=dev)
        t0, tot = time.time(), []
        for i in range(0, len(perm) - args.batch + 1, args.batch):
            loss, parts = losses(model, T, perm[i:i + args.batch], adm, args.w_phys)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(va["x"]), args.batch):
                l, _ = losses(model, V, torch.arange(i, min(i + args.batch, len(va["x"])), device=dev), adm, 0)
                vl.append(l.item())
        rec = {"epoch": ep, "train": float(np.mean(tot)) if tot else None, "val": float(np.mean(vl)),
               "sec": round(time.time() - t0, 1)}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        if rec["val"] < best:
            best = rec["val"]
            torch.save({"model": model.state_dict(), "stats": stats.state(),
                        "config": {"hidden": args.hidden, "layers": args.layers,
                                   "n_node_feat": T["x"].shape[-1], "n_edge_feat": T["edge_attr"].shape[-1]}},
                       out / "best.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print("best val", best)


if __name__ == "__main__":
    main()

"""Minimal MATPOWER .m reader/writer (bus, gen, branch, gencost, bus_name, genfuel).

The ACTIVSg70k case is ~19 MB of plain text; this reads the numeric blocks with
numpy and the cell arrays with a regex, which is enough for this project.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

BUS_COLS = ["bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", "baseKV",
            "zone", "Vmax", "Vmin"]
GEN_COLS = ["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin"]
BRANCH_COLS = ["fbus", "tbus", "r", "x", "b", "rateA", "rateB", "rateC", "ratio", "angle",
               "status", "angmin", "angmax"]


def _block(text: str, name: str) -> str:
    m = re.search(rf"mpc\.{name}\s*=\s*([\[{{])(.*?)[\]}}];", text, re.S)
    if not m:
        raise KeyError(name)
    return m.group(2)


def _numeric(text: str, name: str) -> np.ndarray:
    rows = []
    for line in _block(text, name).splitlines():
        line = line.split("%")[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(v) for v in line.split()])
    return np.array(rows)


def _cells(text: str, name: str) -> list[str]:
    return re.findall(r"'([^']*)'", _block(text, name))


def read_case(path: str | Path) -> dict:
    text = Path(path).read_text()
    case = {
        "baseMVA": float(re.search(r"mpc\.baseMVA\s*=\s*([\d.]+)", text).group(1)),
        "bus": _numeric(text, "bus"),
        "gen": _numeric(text, "gen"),
        "branch": _numeric(text, "branch"),
    }
    for opt, fn in [("gencost", _numeric), ("bus_name", _cells), ("genfuel", _cells),
                    ("gentype", _cells)]:
        try:
            case[opt] = fn(text, opt)
        except KeyError:
            pass
    return case


def write_case(case: dict, path: str | Path, name: str, header: str = "") -> None:
    def fmt(arr):
        return "\n".join("\t" + "\t".join(f"{v:.10g}" for v in row) + ";" for row in arr)

    out = [f"function mpc = {name}"]
    out += ["%" + ln for ln in header.splitlines()]
    out += ["mpc.version = '2';", f"mpc.baseMVA = {case['baseMVA']:g};",
            "mpc.bus = [", fmt(case["bus"]), "];",
            "mpc.gen = [", fmt(case["gen"]), "];",
            "mpc.branch = [", fmt(case["branch"]), "];"]
    if "gencost" in case:
        out += ["mpc.gencost = [", fmt(case["gencost"]), "];"]
    for cell in ("bus_name", "genfuel"):
        if cell in case:
            out += [f"mpc.{cell} = {{"] + [f"\t'{s}';" for s in case[cell]] + ["};"]
    Path(path).write_text("\n".join(out) + "\n")

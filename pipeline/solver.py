"""AC power flow on the Kentucky cut, straight from the MATPOWER arrays.

Newton-Raphson in polar form with the Jacobian from pandapower's pypower port
(`dSbus_dV`). We work on the MATPOWER arrays rather than a converted pandapower
table model because `from_ppc` misplaces the tap side of some ACTIVSg
transformers (bus voltages off by up to 0.06 pu at the stored solution); this
solver reproduces the stored ACTIVSg70k solution to 1e-3 pu.

Modelling choices (see README "Method & limitations"):
  * Generator Q limits are not enforced (PV buses hold voltage).
  * Load changes are balanced first by pro-rata redispatch of Kentucky units
    (capped at Pmax), then by the boundary equivalents in proportion to their
    tie-line ratings; the single slack bus only picks up the residual (losses).
  * An outage that islands buses drops them (their load is reported as lost).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from pandapower.pypower.dSbus_dV import dSbus_dV

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "data/grid"
WEST_LON = -86.8  # load buses west of this follow the TVA demand shape, the rest LGEE


@dataclass
class Result:
    converged: bool
    vm: np.ndarray            # per bus (pu); NaN for dropped (islanded) buses
    va: np.ndarray            # per bus (deg)
    p_from: np.ndarray        # per branch (MW, from end)
    q_from: np.ndarray
    s_from: np.ndarray        # MVA at from end
    s_to: np.ndarray
    loading: np.ndarray       # % of rateA (max of both ends); NaN if unrated/out
    lost_load_mw: float = 0.0
    iterations: int = 0
    slack_p: float = 0.0


@dataclass
class KyGrid:
    buses: pd.DataFrame
    branches: pd.DataFrame
    gens: pd.DataFrame
    base_mva: float = 100.0
    # derived
    n: int = 0
    f: np.ndarray = field(default=None)
    t: np.ndarray = field(default=None)

    @classmethod
    def load(cls) -> "KyGrid":
        g = cls(pd.read_csv(GRID / "buses.csv"), pd.read_csv(GRID / "branches.csv"),
                pd.read_csv(GRID / "gens.csv"))
        g._prepare()
        return g

    def _prepare(self):
        b, br, gn = self.buses, self.branches, self.gens
        self.n = len(b)
        idx = {v: i for i, v in enumerate(b.bus_id)}
        self.f = br.fbus.map(idx).values
        self.t = br.tbus.map(idx).values
        self.gbus = gn.bus.map(idx).values
        self.gon = gn.status.values > 0
        self.internal = (b.role == "internal").values
        self.ref = int(np.where(b.type.values == 3)[0][0])
        self.is_eq_gen = (gn.fuel == "equivalent").values
        # branch series/shunt parameters
        ys = 1 / (br.r.values + 1j * br.x.values)
        tap = np.where(br.ratio.values == 0, 1, br.ratio.values) * np.exp(1j * np.deg2rad(br.angle.values))
        ytt = ys + 1j * br.b.values / 2
        self.yff = ytt / (tap * np.conj(tap))
        self.yft = -ys / np.conj(tap)
        self.ytf = -ys / tap
        self.ytt = ytt
        self.ysh = np.zeros(self.n, complex)  # equivalent shunts were dropped at extraction
        self.rate = br.rateA.values.astype(float)
        self.rated = (self.rate > 0) & (br.role.values != "equivalent")
        # bus voltage setpoints from generators
        self.vset = np.ones(self.n)
        self.vset[self.gbus[self.gon]] = gn.Vg.values[self.gon]
        self.pv_bus = np.zeros(self.n, bool)
        self.pv_bus[self.gbus[self.gon]] = True
        # load region (0 = LGEE shape, 1 = TVA shape) and base injections
        self.region = (b.lon.values < WEST_LON).astype(int)
        self.pd0 = b.pd_mw.values.copy()
        self.qd0 = b.qd_mvar.values.copy()
        self.pg0 = gn.Pg.values.copy()
        self.qg0 = gn.Qg.values.copy()
        self.pmax = gn.Pmax.values.copy()
        # boundary participation: tie-line rating at each boundary bus
        tie = (br.role == "tie").values
        w = np.zeros(self.n)
        np.add.at(w, self.f[tie], self.rate[tie])
        np.add.at(w, self.t[tie], self.rate[tie])
        self.eq_weight = np.where(self.is_eq_gen, w[self.gbus], 0.0)
        self.eq_weight /= self.eq_weight.sum()

    # ------------------------------------------------------------------
    def ybus(self, on: np.ndarray, cap_factor: float = 1.0) -> csr_matrix:
        f, t = self.f[on], self.t[on]
        Y = csr_matrix((np.r_[self.yff[on], self.yft[on], self.ytf[on], self.ytt[on]],
                        (np.r_[f, f, t, t], np.r_[f, t, f, t])), shape=(self.n, self.n))
        sh = self.bus_shunt_y().copy()
        cap = (sh.imag > 0) & self.internal
        sh[cap] = sh[cap].real + 1j * sh[cap].imag * cap_factor
        return (Y + csr_matrix((sh, (np.arange(self.n), np.arange(self.n))), shape=(self.n, self.n))).tocsr()

    def bus_shunt_y(self):
        if not hasattr(self, "_bsh"):
            from matpower_io import read_case
            c = read_case(GRID / "ky_case.m")
            self._bsh = (c["bus"][:, 4] + 1j * c["bus"][:, 5]) / self.base_mva
        return self._bsh

    def dispatch(self, load_scale: np.ndarray):
        """Per-bus load scale -> (pd, qd, pg) with pro-rata internal redispatch and
        boundary participation for whatever Kentucky units cannot cover."""
        pd_ = self.pd0 * load_scale
        qd_ = self.qd0 * load_scale
        dP = pd_[self.internal].sum() - self.pd0[self.internal].sum()
        pg = self.pg0.copy()
        ig = ~self.is_eq_gen & self.gon
        ratio = pd_[self.internal].sum() / self.pd0[self.internal].sum()
        pg[ig] = np.clip(self.pg0[ig] * ratio, 0, self.pmax[ig])
        covered = pg[ig].sum() - self.pg0[ig].sum()
        pg += (dP - covered) * self.eq_weight
        return pd_, qd_, pg

    def solve(self, load_scale: np.ndarray | float = 1.0, out_branch: int | None = None,
              v0: np.ndarray | None = None, tol: float = 1e-6, max_it: int = 25) -> Result:
        load_scale = np.broadcast_to(np.asarray(load_scale, float), (self.n,))
        pd_, qd_, pg = self.dispatch(load_scale)
        on = self.branches.status.values > 0
        if out_branch is not None:
            on = on.copy()
            on[out_branch] = False
        # islands: keep only the component that holds the slack bus
        A = csr_matrix((np.ones(on.sum()), (self.f[on], self.t[on])), shape=(self.n, self.n))
        _, lab = connected_components(A, directed=False)
        alive = lab == lab[self.ref]
        lost = float(pd_[~alive & self.internal].sum())
        on = on & alive[self.f] & alive[self.t]

        # switched-capacitor proxy: shunt capacitors in Kentucky are switched out
        # as load falls (all in at >= 100% of base load, all out at <= 40%)
        mean_scale = float(pd_[self.internal].sum() / self.pd0[self.internal].sum())
        Y = self.ybus(on, cap_factor=float(np.clip((mean_scale - 0.4) / 0.6, 0, 1)))
        Sg = np.zeros(self.n, complex)
        gmask = self.gon & alive[self.gbus]
        np.add.at(Sg, self.gbus[gmask], pg[gmask] + 1j * self.qg0[gmask])
        Sbus = (Sg - (pd_ + 1j * qd_)) / self.base_mva
        pvb = self.pv_bus & alive
        pvb[self.ref] = False
        pv = np.where(pvb)[0]
        pq = np.where(alive & ~self.pv_bus)[0]
        pq = pq[pq != self.ref]
        pvpq = np.r_[pv, pq]

        V = (v0.copy() if v0 is not None else
             np.where(self.pv_bus, self.vset, 1.0) * np.exp(1j * np.deg2rad(self.buses.va_base_deg.values)))
        V[self.pv_bus] = self.vset[self.pv_bus] * V[self.pv_bus] / np.abs(V[self.pv_bus])
        V[~alive] = 0
        Va, Vm = np.angle(V), np.abs(V)
        conv, it = False, 0
        for it in range(max_it + 1):
            mis = V * np.conj(Y @ V) - Sbus
            F = np.r_[mis[pvpq].real, mis[pq].imag]
            if np.max(np.abs(F)) < tol:
                conv = True
                break
            if it == max_it or not np.isfinite(F).all():
                break
            dS_dVm, dS_dVa = dSbus_dV(Y, V)
            J11 = dS_dVa[pvpq][:, pvpq].real
            J12 = dS_dVm[pvpq][:, pq].real
            J21 = dS_dVa[pq][:, pvpq].imag
            J22 = dS_dVm[pq][:, pq].imag
            from scipy.sparse import bmat
            J = bmat([[J11, J12], [J21, J22]], format="csc")
            dx = spsolve(J, -F)
            Va[pvpq] += dx[:len(pvpq)]
            Vm[pq] += dx[len(pvpq):]
            V = Vm * np.exp(1j * Va)
        # branch flows
        If = self.yff * V[self.f] + self.yft * V[self.t]
        It = self.ytf * V[self.f] + self.ytt * V[self.t]
        Sf = V[self.f] * np.conj(If) * self.base_mva
        St = V[self.t] * np.conj(It) * self.base_mva
        Sf[~on] = 0
        St[~on] = 0
        smax = np.maximum(np.abs(Sf), np.abs(St))
        loading = np.where(self.rated & on, 100 * smax / np.where(self.rate > 0, self.rate, 1), np.nan)
        vm = np.where(alive, np.abs(V), np.nan)
        va = np.where(alive, np.rad2deg(np.angle(V)), np.nan)
        slack = (V * np.conj(Y @ V))[self.ref] * self.base_mva
        return Result(conv, vm, va, Sf.real, Sf.imag, np.abs(Sf), np.abs(St), loading, lost, it,
                      float(slack.real + pd_[self.ref]))

    def load_scale_vector(self, scale_lgee: float, scale_tva: float) -> np.ndarray:
        return np.where(self.region == 1, scale_tva, scale_lgee)

"""Hourly load scales and weather for the Kentucky cut.

Scale convention: 1.0 = the ACTIVSg70k base load in area 27 (17.4 GW), taken to
be the coincident peak. Each region's hourly scale is EIA-930 demand divided by
that balancing area's 2023-2024 maximum, so the scale is <= 1 for every
observed hour. LGEE drives central/eastern buses, TVA drives buses west of
86.8 W (see solver.WEST_LON).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data/inputs"
WX_POINTS = {
    "louisville": (38.25, -85.76), "lexington": (38.04, -84.50), "bowling_green": (36.99, -86.44),
    "paducah": (37.08, -88.60), "pikeville": (37.48, -82.52),
}


def hourly_scales() -> pd.DataFrame:
    d = pd.read_parquet(INP / "eia930_ky.parquet").dropna()
    # drop obvious EIA-930 glitches (spikes > 4 sigma from a 24 h rolling median)
    for c in d.columns:
        med = d[c].rolling(25, center=True, min_periods=5).median()
        bad = (d[c] - med).abs() > 4 * d[c].std()
        d.loc[bad, c] = np.nan
    d = d.interpolate(limit=3).dropna()
    s = pd.DataFrame({"lgee": d.LGEE / d.LGEE.max(), "tva": d.TVA / d.TVA.max(),
                      "lgee_mw": d.LGEE, "tva_mw": d.TVA}, index=d.index)
    w = weather()
    if w is not None:
        s = s.join(w, how="left")
    return s


def weather() -> pd.DataFrame | None:
    fn = INP / "era5_ky.parquet"
    if not fn.exists():
        return None
    w = pd.read_parquet(fn)
    w.index = pd.to_datetime(w.index)
    if w.index.tz is not None:      # EIA-930 index is naive UTC
        w.index = w.index.tz_convert("UTC").tz_localize(None)
    return w


def align_index(s: pd.DataFrame) -> pd.DataFrame:
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    return s


def nearest_point(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    names = list(WX_POINTS)
    P = np.array([WX_POINTS[k] for k in names])
    d = (lat[:, None] - P[None, :, 0]) ** 2 + ((lon[:, None] - P[None, :, 1]) * 0.8) ** 2
    return np.array(names)[np.argmin(d, axis=1)]


def representative_hours(s: pd.DataFrame) -> list[dict]:
    """Load levels offered in the what-if dashboard."""
    tot = s.lgee_mw + s.tva_mw
    summer = tot[tot.index.month.isin([6, 7, 8, 9])]
    winter = tot[tot.index.month.isin([12, 1, 2])]
    spring = tot[tot.index.month.isin([3, 4, 5, 10])]
    picks = [
        ("light", "Light load: spring night minimum", spring.idxmin()),
        ("typical", "Typical: median hour", (tot - tot.median()).abs().idxmin()),
        ("winter_peak", "Winter peak hour", winter.idxmax()),
        ("summer_peak", "Summer peak hour", summer.idxmax()),
    ]
    out = []
    for key, label, t in picks:
        row = s.loc[t]
        out.append({"key": key, "label": label, "time_utc": t.isoformat(),
                    "lgee": float(row.lgee), "tva": float(row.tva),
                    "lgee_mw": float(row.lgee_mw), "tva_mw": float(row.tva_mw),
                    "weather": {c: round(float(row[c]), 1) for c in s.columns
                                if c.startswith(("temperature", "wind")) and pd.notna(row[c])}})
    # stress cases: summer peak with Kentucky load scaled up (what-if, not observed)
    sp = out[-1]
    for k, f in [("stress110", 1.10), ("stress120", 1.20)]:
        out.append({**sp, "key": k, "label": f"Stress: summer peak +{round((f-1)*100)}% (hypothetical)",
                    "lgee": sp["lgee"] * f, "tva": sp["tva"] * f,
                    "lgee_mw": sp["lgee_mw"] * f, "tva_mw": sp["tva_mw"] * f})
    return out

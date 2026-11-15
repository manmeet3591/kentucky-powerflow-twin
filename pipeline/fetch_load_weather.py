"""Hourly demand (EIA-930) and weather (ERA5 via Open-Meteo) for the Kentucky region.

EIA-930 six-month BALANCE files need no API key. We keep two balancing areas:
  LGEE  Louisville Gas & Electric / Kentucky Utilities (central + north KY)
  TVA   Tennessee Valley Authority (serves western and southern KY)
ERA5 hourly 2 m temperature and 10 m wind come from the Open-Meteo archive API
(no key) for a few KY points. Open-Meteo rate-limits hourly, so each point is
cached and the script can simply be re-run.

Outputs: data/inputs/eia930_ky.parquet, data/inputs/era5_ky.parquet
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/eia930"
WX = ROOT / "data/raw/era5"
OUT = ROOT / "data/inputs"
YEARS = [2023, 2024]
BAS = ["LGEE", "TVA"]
POINTS = {  # name: (lat, lon)
    "louisville": (38.25, -85.76),
    "lexington": (38.04, -84.50),
    "bowling_green": (36.99, -86.44),
    "paducah": (37.08, -88.60),
    "pikeville": (37.48, -82.52),
}


def fetch_eia():
    RAW.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in YEARS:
        for half in ["Jan_Jun", "Jul_Dec"]:
            fn = RAW / f"EIA930_BALANCE_{y}_{half}.csv"
            if not fn.exists():
                url = f"https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/{fn.name}"
                print("download", url)
                r = requests.get(url, timeout=600)
                r.raise_for_status()
                fn.write_bytes(r.content)
            df = pd.read_csv(fn, low_memory=False, thousands=",")
            df = df[df["Balancing Authority"].isin(BAS)]
            col = [c for c in df.columns if c.startswith("Demand (MW)") and "Adjusted" in c] or \
                  [c for c in df.columns if c.startswith("Demand (MW)")]
            t = pd.to_datetime(df["UTC Time at End of Hour"], format="mixed")
            frames.append(pd.DataFrame({"time_utc": t, "ba": df["Balancing Authority"],
                                        "demand_mw": pd.to_numeric(df[col[0]], errors="coerce")}))
    d = pd.concat(frames).pivot_table(index="time_utc", columns="ba", values="demand_mw")
    d = d.sort_index().interpolate(limit=6)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "eia930_ky.parquet")
    print("EIA-930:", d.shape, d.describe().loc[["min", "mean", "max"]].round(0).to_dict())


def fetch_era5():
    WX.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, (lat, lon) in POINTS.items():
        fn = WX / f"{name}.csv"
        if not fn.exists():
            url = ("https://archive-api.open-meteo.com/v1/archive"
                   f"?latitude={lat}&longitude={lon}&start_date={YEARS[0]}-01-01&end_date={YEARS[-1]}-12-31"
                   "&hourly=temperature_2m,wind_speed_10m&models=era5&timezone=UTC&wind_speed_unit=ms")
            r = requests.get(url, timeout=120)
            js = r.json()
            if js.get("error"):
                print(f"Open-Meteo refused {name}: {js.get('reason')}; re-run later")
                continue
            pd.DataFrame(js["hourly"]).to_csv(fn, index=False)
            time.sleep(2)
        w = pd.read_csv(fn, parse_dates=["time"])
        w["point"] = name
        frames.append(w)
    if len(frames) < len(POINTS):
        return False
    w = pd.concat(frames).pivot_table(index="time", columns="point",
                                      values=["temperature_2m", "wind_speed_10m"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w.index.name = "time_utc"
    w.to_parquet(OUT / "era5_ky.parquet")
    print("ERA5:", w.shape)
    return True


if __name__ == "__main__":
    if "--weather-only" not in sys.argv:
        fetch_eia()
    ok = fetch_era5()
    sys.exit(0 if ok else 2)

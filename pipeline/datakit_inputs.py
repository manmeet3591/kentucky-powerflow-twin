"""Inputs for IBM gridfm-datakit on the Kentucky case.

  * data/grid/datakit/ky_case.m : the reduced case without the cell arrays
    (bus_name, genfuel) that some MATPOWER readers reject.
  * <gridfm_datakit>/load_profiles/lgee_2023_2024.csv : EIA-930 LGEE hourly
    demand, one value per line, which datakit's `agg_load_profile` generator
    min-max scales into global load factors.
"""
from importlib import resources
from pathlib import Path

import pandas as pd

from matpower_io import read_case, write_case

ROOT = Path(__file__).resolve().parents[1]

c = read_case(ROOT / "data/grid/ky_case.m")
for k in ("bus_name", "genfuel", "gentype"):
    c.pop(k, None)
out = ROOT / "data/grid/datakit"
out.mkdir(parents=True, exist_ok=True)
write_case(c, out / "ky_case.m", "ky_case", header=" Kentucky cut of ACTIVSg70k (synthetic); see README")

d = pd.read_parquet(ROOT / "data/inputs/eia930_ky.parquet")["LGEE"].dropna()
prof = Path(resources.files("gridfm_datakit.load_profiles")) / "lgee_2023_2024.csv"
d.round(1).to_csv(prof, index=False, header=False)
print("wrote", out / "ky_case.m", "and", prof, len(d), "hours")

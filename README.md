# Kentucky Power Flow Twin

A prototype AI power flow digital twin for a Kentucky-region transmission grid, built for the Disaster Science Operations Center (DSOC). Layer 6 of "where AI shows up in a utility" covers grid operations and digital twins. This repo covers the hourly AC power flow, the network twin and the N-1 what-if studies, plus a graph neural network (GNN) surrogate meant to answer the same questions in milliseconds.

**Dashboard:** `docs/` (static, GitHub Pages ready). Pick a load level and take a line out to see branch loadings and voltage violations. Every result shown today comes from the AC solver and is precomputed. The surrogate toggle stays off until the GNN has been trained on TACC.

> **Synthetic grid.** The network is the Kentucky footprint (area 27) of the Texas A&M ACTIVSg70k synthetic Eastern US case. It contains no CEII and models none of Kentucky's actual lines, substations or utilities. Nothing here describes the real grid.

## What the grid is, and is not

| | |
|---|---|
| Source | TAMU ACTIVSg70k, MATPOWER `data/case_ACTIVSg70k.m` (Birchfield et al., IEEE TPWRS 32(4), 2017). Fully synthetic, no CEII. |
| Kentucky cut | Area 27: 85% of its bus names match Kentucky places in the Census gazetteer, the highest share of any area (`pipeline/extract_kentucky.py`). 1,662 buses, 69–345 kV plus generator terminals, 1,923 branches, 17.4 GW load, 21.6 GW generator Pmax. |
| Outside Kentucky | 68,184 external buses Kron-reduced onto the 154 boundary buses tied to area 27. This gives 595 equivalent branches and voltage-holding equivalent generators (extended Ward). The reduced case reproduces the stored 70k solution within 0.34 MW and 1e-5 pu. Equivalent shunts are dropped, because the Kron row sums imply hundreds of GW. |
| Coordinates | Each bus sits at the centroid of the town in its synthetic name: Census gazetteer for 1,559 buses, the mean of connected buses for 257, clamped to Kentucky. The map is schematic. |
| Ratings | Static rateA from the case. No weather-dependent ratings. |

## Pipeline

```
pipeline/fetch_load_weather.py   EIA-930 LGEE + TVA hourly demand (2023-24), ERA5 temp/wind via Open-Meteo (5 KY points)
pipeline/extract_kentucky.py     ACTIVSg70k -> data/grid/{ky_case.m, buses.csv, branches.csv, gens.csv, summary.json}
pipeline/solver.py               AC Newton-Raphson (pandapower PYPOWER Jacobians) on the MATPOWER arrays; N-1 helper
pipeline/scenarios.py            hourly load scales (1.0 = ACTIVSg base = coincident peak) and weather
pipeline/generate_dataset.py     N-0/N-1 AC samples as npz shards (TACC array jobs)
pipeline/precompute_whatif.py    full N-1 sweep at 6 load levels -> docs/data/
pipeline/datakit_inputs.py       inputs for IBM gridfm-datakit (case + LGEE load profile)
pipeline/check_datakit.py        PowerModels (datakit) solutions vs our network model
surrogate/model.py               edge-conditioned message-passing GNN (Vm, Va, P/Q flows, loading)
surrogate/train.py, evaluate.py  training; held-out evaluation vs AC solver and DC power flow
surrogate/predict_whatif.py      surrogate results for every dashboard case (docs/data/gnn_level_*.json)
jobs/*.slurm                     TACC: generate (CPU), train/evaluate (GPU), datakit, graphkit
```

Local run (CPU, about 20 minutes, no training):

```bash
conda create -n ky-pf-twin python=3.11 && conda activate ky-pf-twin
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
mkdir -p data/raw && curl -L -o data/raw/case_ACTIVSg70k.m https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case_ACTIVSg70k.m
# Census gazetteer (places + county subdivisions, 2023) into data/raw/gazetteer/
python pipeline/fetch_load_weather.py
python pipeline/extract_kentucky.py
python pipeline/precompute_whatif.py 6
python -m http.server -d docs 8767
```

## Modelling choices

- **Load.** LGEE's shape drives central and eastern buses and TVA's drives buses west of 86.8°W. Each region's scale is its hourly demand divided by its 2023–24 maximum. Training samples add ±5% per-bus lognormal noise, and 25% of them get a 1.0–1.25× stress stretch.
- **Dispatch.** Kentucky units scale pro rata, capped at Pmax. The remaining imbalance goes to the boundary equivalents in proportion to their tie ratings, and the slack picks up losses. This is a proxy, not an economic dispatch.
- **Voltage control.** Generator Q limits are not enforced. Shunt capacitors are switched with load by a simple rule: all in at ≥100% load, all out at ≤40%. There is no tap or phase-shifter control.
- **Contingencies.** Every rated Kentucky branch is a contingency (1,473). 169 of them island part of the network; the islanded load is dropped and reported. Overload means >100% of rateA, using the larger of the two ends. A voltage violation is a Kentucky bus at 69 kV or above outside 0.95–1.05 pu, with 0.90–1.10 also reported.
- **Independent check.** `gridfm-datakit` ran locally on the case (8 N-1 scenarios, PowerModels.jl AC PF). Its solutions satisfy our network model with 0.0000 MW active-power mismatch and identical branch flows (`pipeline/check_datakit.py`). Reactive balance differs only at the 97 shunt buses, because datakit reports shunts differently.
- **Why not pandapower's converter.** `from_ppc` put some ACTIVSg transformer taps on the wrong side, which moved bus voltages by up to 0.06 pu at the stored solution. The solver therefore works on the MATPOWER arrays directly.

## N-1 results on the synthetic case (AC solver)

At the summer peak hour, the intact network peaks at 86% loading. Single 345 kV outages around Hopkinsville push neighbouring lines up to 168% and leave 4 overloads. Light-load hours show more high-voltage buses. Every case converged. See the dashboard's *Contingency ranking* tab for every load level.

## Surrogate and evaluation (TACC)

The model is a 12-layer, 128-hidden edge-conditioned GNN with a global context term, about 3.3M parameters. The outaged branch is switched off with an edge mask, and an optional physics loss recomputes flows from the predicted voltages.

- **Data.** Training uses 2023 hours. Testing uses 2024 hours, split into outages seen in training and outages of 130 held-out lines that are never outaged in training.
- **Baselines.** AC Newton-Raphson, which is the reference, and DC power flow.
- **Metrics.** Vm/Va MAE, loading MAE and 95th-percentile error, overload and voltage-violation precision and recall, and ms per scenario.

```bash
# on a TACC login node, once
bash jobs/00_setup_env.sh            # edit -A ALLOCATION in jobs/*.slurm first
sbatch jobs/01_generate.slurm        # 48k AC samples, one CPU node, ~5 min
sbatch jobs/02_train.slurm           # A100 (Lonestar6 gpu-a100) or GH200 (Vista gh)
sbatch jobs/03_evaluate.slurm        # writes docs/data/surrogate_eval.json + gnn_level_*.json
# optional: IBM GridFM tools on the same case
sbatch jobs/04_datakit.slurm         # gridfm-datakit, PowerModels.jl AC PF, random N-1, LGEE profile
sbatch jobs/05_graphkit.slurm        # gridfm-graphkit HGNS on the datakit output
```

Copy `docs/data/surrogate_eval.json`, `docs/data/gnn_level_*.json` and `docs/data/manifest.json` back into the repo and push. The dashboard then switches on its *GNN surrogate* option and the *Surrogate* tab.

Local CPU sanity checks only: 48 samples per split, a 3-epoch run of a 57k-parameter model, then evaluation and dashboard export. These checks show that the code runs; they say nothing about accuracy.

## Limits

- Synthetic grid. The external equivalent is fixed at one operating point. There is no market dispatch, no Q limits, and no dynamics or stability analysis.
- Map positions are approximate town centroids, and lines are drawn as straight segments.
- The surrogate is a screening tool. Flagged cases should be re-run in the AC solver.

## Data and credits

- TAMU ACTIVSg70k via MATPOWER (BSD), EIA-930 (public domain), ERA5 (Copernicus/ECMWF) via the Open-Meteo archive (CC BY 4.0), and the US Census Bureau 2023 gazetteer.
- IBM / LF Energy `gridfm-datakit` and `gridfm-graphkit` (Apache-2.0), both presented at GridFM Harvard 2026. Notes: https://manmeet3591.github.io/learnings_from_gridfm.html

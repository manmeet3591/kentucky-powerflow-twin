#!/bin/bash
#SBATCH -J kpt-datakit
#SBATCH -p skx
#SBATCH -N 1 -n 48
#SBATCH -t 06:00:00
#SBATCH -A ALLOCATION
#SBATCH -o logs/%x-%j.out
# Independent dataset with IBM gridfm-datakit (PowerModels.jl AC PF, N-1 branch/gen outages)
# on the same Kentucky case, driven by the LGEE 2023-24 hourly shape. Output is the
# GridFM parquet format that gridfm-graphkit trains on (jobs/05_graphkit.slurm).
source jobs/common.sh
python pipeline/datakit_inputs.py      # writes the LGEE profile into datakit's load_profiles/
gridfm_datakit generate configs/datakit_ky.yaml

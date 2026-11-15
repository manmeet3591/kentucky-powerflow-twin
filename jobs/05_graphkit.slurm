#!/bin/bash
#SBATCH -J kpt-graphkit
#SBATCH -p gpu-a100
#SBATCH -N 1 -n 1
#SBATCH -t 08:00:00
#SBATCH -A ALLOCATION
#SBATCH -o logs/%x-%j.out
# Optional comparison: train IBM gridfm-graphkit's GNN on the datakit output and, if wanted,
# fine-tune the public GENCO PF checkpoint (huggingface.co/gridfm/genco-pf-contingency).
# graphkit's CLI/config keys change between releases: check `gridfm_graphkit --help` and
# the repo's examples/config/*PF_datakit*.yaml against configs/graphkit_ky.yaml first.
source jobs/common.sh
gridfm_graphkit train --config configs/graphkit_ky.yaml --data_path "$KPT_DATA/datakit"
gridfm_graphkit evaluate --config configs/graphkit_ky.yaml --data_path "$KPT_DATA/datakit"

#!/bin/bash
#SBATCH -J kpt-eval
#SBATCH -p gpu-a100
#SBATCH -N 1 -n 1
#SBATCH -t 01:00:00
#SBATCH -A ALLOCATION
#SBATCH -o logs/%x-%j.out
# Held-out evaluation against the AC solver (2024 hours; seen and never-seen outages),
# DC power flow baseline, timing on GPU and CPU; then surrogate results for the dashboard.
source jobs/common.sh
python surrogate/evaluate.py --ckpt runs/gnn/best.pt --data "$KPT_DATA/dataset"
python surrogate/evaluate.py --ckpt runs/gnn/best.pt --data "$KPT_DATA/dataset" --device cpu --no_docs \
  > runs/gnn/eval_cpu.log
python surrogate/predict_whatif.py --ckpt runs/gnn/best.pt
echo "copy back: docs/data/surrogate_eval.json docs/data/gnn_level_*.json runs/gnn/{best.pt,eval.json,train_log.json}"

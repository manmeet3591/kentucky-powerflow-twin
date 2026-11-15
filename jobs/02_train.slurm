#!/bin/bash
#SBATCH -J kpt-train
#SBATCH -p gpu-a100
#SBATCH -N 1 -n 1
#SBATCH -t 08:00:00
#SBATCH -A ALLOCATION
#SBATCH -o logs/%x-%j.out
# GNN surrogate (surrogate/model.py). 40k samples x 2 GB float32 fit in A100/GH200 memory.
source jobs/common.sh
python surrogate/train.py --data "$KPT_DATA/dataset" --out runs/gnn \
  --epochs ${EPOCHS:-150} --batch 16 --hidden 128 --layers 12 --lr 1e-3 --w_phys 0.1

#!/bin/bash
#SBATCH -J kpt-gen
#SBATCH -p skx
#SBATCH -N 1 -n 48
#SBATCH -t 02:00:00
#SBATCH -A ALLOCATION
#SBATCH -o logs/%x-%j.out
# AC power flow samples with pandapower-pypower Newton-Raphson (pipeline/solver.py).
# One node runs 48 shards in parallel: 40 train, 4 val, 2 test_seen, 2 test_unseen,
# 1000 samples each (~0.15 s/sample -> ~3 min per shard). Total 48k samples.
source jobs/common.sh
N=${N:-1000}
run() { python pipeline/generate_dataset.py --split $1 --n $N --shard $2 > logs/gen_$1_$2.log 2>&1; }
for s in $(seq 0 39); do run train $s & done
for s in 0 1 2 3;   do run val $s & done
for s in 0 1;       do run test_seen $s & done
for s in 0 1;       do run test_unseen $s & done
wait
mv data/dataset/*.npz "$KPT_DATA/dataset/" 2>/dev/null
ls "$KPT_DATA/dataset" | wc -l

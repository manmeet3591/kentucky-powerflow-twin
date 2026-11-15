# Shared settings for kentucky-powerflow-twin jobs on TACC. Edit ALLOCATION (in each
# job's #SBATCH -A line) and these paths once.
# CPU data generation: Stampede3 (partition skx) or Frontera (normal) or Lonestar6 (normal).
# GPU training: Lonestar6 (partition gpu-a100) or Vista (partition gh).
export KPT_HOME=${KPT_HOME:-$WORK/kentucky-powerflow-twin}   # git clone here
export KPT_DATA=${KPT_DATA:-$SCRATCH/kpt-data}               # generated samples (large)
export KPT_ENV=${KPT_ENV:-$WORK/envs/kpt}                    # conda env made by 00_setup_env.sh
source $WORK/miniforge3/etc/profile.d/conda.sh 2>/dev/null || module load conda 2>/dev/null
conda activate "$KPT_ENV"
cd "$KPT_HOME"
mkdir -p "$KPT_DATA/dataset" logs runs
# keep each solver process single-threaded; parallelism comes from tasks/arrays
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

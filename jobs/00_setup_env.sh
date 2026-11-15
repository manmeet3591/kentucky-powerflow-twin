# Run once on a TACC login node.
if [ ! -d $WORK/miniforge3 ]; then
  curl -sL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -o /tmp/mf.sh
  bash /tmp/mf.sh -b -p $WORK/miniforge3
fi
source $WORK/miniforge3/etc/profile.d/conda.sh
conda create -y -p $WORK/envs/kpt python=3.11
conda activate $WORK/envs/kpt
# GPU torch build for Lonestar6/Vista; use the cu12x wheel that matches the node driver
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r $WORK/kentucky-powerflow-twin/requirements.txt
# optional IBM GridFM tools (datakit pulls Julia + PowerModels on first use)
pip install gridfm-datakit
pip install gridfm-graphkit   # needs torch-scatter/torch-sparse wheels matching torch+CUDA

#!/bin/bash -l
#SBATCH --job-name="spmm"
#SBATCH --account="g34"
#SBATCH --time=00:30:00
#SBATCH --nodes=128
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=normal
#SBATCH --constraint=gpu

module load daint-gpu
module load cudatoolkit
module swap PrgEnv-cray PrgEnv-gnu

# Setup the compiler
#
export CC=`which cc`
export CXX=`which CC`
export FC=`which ftn`

export CRAYPE_LINK_TYPE=dynamic

# Enable threading
# 
export MKL_NUM_THREADS=12

# Set virtual environment
export VENV_ROOT="${HOME}/py3.11spmm"
source ${VENV_ROOT}/bin/activate
export PATH="${VENV_ROOT}/bin:$PATH"

srun --contiguous python -OO ../../spmm2d_distr.py --validate False -f $1 -k $2 --columns=$3 --device='gpu' -d file --iterations 7
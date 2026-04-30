#!/bin/bash
#
# Run the Quimp-trained Cellpose model on BOTH batches (60 cells total).
# Outputs go to parallel directories; existing self-trained outputs untouched.
#

#SBATCH --job-name=quimp_inference
#SBATCH --output=/dcs/pg25/u1898019/Desktop/Categorised_Data/quimp_inf_%j.out
#SBATCH --error=/dcs/pg25/u1898019/Desktop/Categorised_Data/quimp_inf_%j.err
#SBATCH --partition=falcon,gecko
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

echo "[Slurm] JobID: ${SLURM_JOB_ID:-NA}"
echo "[Slurm] Host:  $(hostname)"
echo "[Slurm] Date:  $(date)"

set +u
source /etc/profile.d/conda.sh 2>/dev/null || true
if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
    module load CUDA 2>/dev/null || true
fi
conda activate dgac
set -u

cd /dcs/pg25/u1898019/Desktop/Categorised_Data

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python process_with_quimp.py

echo "[Done] $(date)"

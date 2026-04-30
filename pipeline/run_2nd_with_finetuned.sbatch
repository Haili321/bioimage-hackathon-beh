#!/bin/bash
#
# Run the fine-tuned Cellpose pipeline on the 2nd-upload batch.
# Generalisation test: same model, fresh cells.
#

#SBATCH --job-name=hack_2nd_finetune
#SBATCH --output=/dcs/pg25/u1898019/Desktop/Categorised_Data/finetune_2nd_%j.out
#SBATCH --error=/dcs/pg25/u1898019/Desktop/Categorised_Data/finetune_2nd_%j.err
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

python process_2nd_with_finetuned.py

echo "[Done] $(date)"

#!/bin/bash
#
# Fine-tune Cellpose cyto3 on our pseudo-ground-truth training data
# (filtered v3 outputs from Categorised_Data, 586 image-mask pairs).
#
# Usage:
#   sbatch run_finetune.sbatch
#

#SBATCH --job-name=cyto3_finetune
#SBATCH --output=/dcs/pg25/u1898019/Desktop/Categorised_Data/finetune_%j.out
#SBATCH --error=/dcs/pg25/u1898019/Desktop/Categorised_Data/finetune_%j.err
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

cd /dcs/pg25/u1898019/Desktop/Categorised_Data_centered/training_data

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

python -W ignore -m cellpose --train \
    --dir /dcs/pg25/u1898019/Desktop/Categorised_Data_centered/training_data \
    --pretrained_model cyto3 \
    --img_filter _img \
    --mask_filter _masks \
    --n_epochs 100 \
    --learning_rate 0.05 \
    --weight_decay 0.0001 \
    --batch_size 8 \
    --min_train_masks 1 \
    --use_gpu \
    --verbose

echo "[Done] $(date)"

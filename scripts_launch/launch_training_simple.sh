#!/bin/bash
# Simple launch script for quick testing - launches just a few jobs

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Quick test: launch one diffusion and one flow matching job
echo "Launching test jobs..."

# Diffusion job
sbatch scripts_launch/job_training.sh \
    --model diffusion \
    --seed 0 \
    --lr 1e-4 \
    --batch-size 256 \
    --max-demos 100 \
    --epochs 10 \
    --wandb \
    --wandb-project sd-pusht

# Flow matching job
sbatch scripts_launch/job_training.sh \
    --model flowmatching \
    --seed 0 \
    --lr 1e-4 \
    --batch-size 256 \
    --max-demos 100 \
    --epochs 10 \
    --wandb \
    --wandb-project sd-pusht

echo "Test jobs launched. Check logs_berzelius/ for output."


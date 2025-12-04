#!/bin/bash
# Simple launch script for quick testing - launches local flow 2D policy training jobs
# One with GT reference frame, one without

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

echo "Launching test jobs for local flow 2D policy..."

# Job with ground truth reference frame (oracle mode)
echo "Launching job WITH ground truth reference frame..."
sbatch scripts_launch/job_training_local_flow_2d.sh \
    --seed 0 \
    --lr 1e-4 \
    --batch-size 256 \
    --max-demos 50 \
    --epochs 100 \
    --eval-every 10 \
    --wandb \
    --wandb-project sd-pusht-local-flow-2d \
    --use-gt-reference \
    --use-position-decoder \
    --share-noise \
    --position-loss-coeff 1.0

# Job without ground truth reference frame (learned mode)
echo "Launching job WITHOUT ground truth reference frame (learned)..."
sbatch scripts_launch/job_training_local_flow_2d.sh \
    --seed 0 \
    --lr 1e-4 \
    --batch-size 256 \
    --max-demos 50 \
    --epochs 100 \
    --eval-every 10 \
    --wandb \
    --wandb-project sd-pusht-local-flow-2d \
    --use-position-decoder \
    --share-noise \
    --position-loss-coeff 1.0

echo "Test jobs launched. Check logs_berzelius/ for output."


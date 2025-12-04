#!/bin/bash
# Simple launch script for quick evaluation testing

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Example: Evaluate a single checkpoint
# Modify these paths to match your experiments
CKPT_PATH="log/dp/sd-pusht-diffusion-demos-100-seed1234/checkpoints/best_ema_model.pt"

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found: $CKPT_PATH"
    echo "Please update CKPT_PATH in this script to point to a valid checkpoint."
    exit 1
fi

echo "Launching evaluation for: $CKPT_PATH"

sbatch scripts_launch/job_eval.sh \
    --ckpt-path "$CKPT_PATH" \
    --num-envs 64 \
    --max-steps 300

echo "Evaluation job launched. Check logs_berzelius/ for output."


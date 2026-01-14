#!/bin/bash
# Launch evaluation jobs for different trained local flow 2D models

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Base directory for experiments
BASE_DIR="log/dp"

# Define which experiments to evaluate
# You can customize these arrays to match your training experiments
SEEDS=(1234)
MAX_DEMOS=(20 60 100 140 180 200)
USE_GT_REFERENCE_OPTIONS=(true)
SHARE_NOISE_OPTIONS=(false)

# Checkpoint types to evaluate (final model and/or best model)
CHECKPOINT_TYPES=("best_ema_model.pt")

# Evaluation parameters (can be overridden per job)
NUM_ENVS=128
MAX_STEPS=300

# Launch jobs for each combination
for seed in "${SEEDS[@]}"; do
    for max_demos in "${MAX_DEMOS[@]}"; do
        for use_gt_ref in "${USE_GT_REFERENCE_OPTIONS[@]}"; do
            for share_noise in "${SHARE_NOISE_OPTIONS[@]}"; do
                for ckpt_type in "${CHECKPOINT_TYPES[@]}"; do
                    # Construct experiment name (matching train_local_flow_2d.py)
                    if [ "$share_noise" = true ]; then
                        noise_str="share-noise"
                    else
                        noise_str="no-share-noise"
                    fi
                    if [ "$use_gt_ref" = true ]; then
                        ref_str="gt-ref"
                    else
                        ref_str="pred-ref"
                    fi
                    exp_name="sd-pusht-local-flow-2d_V3-demos-${max_demos}-seed${seed}-${noise_str}-${ref_str}"
                    ckpt_path="${BASE_DIR}/${exp_name}/checkpoints/${ckpt_type}"
                    
                    # Check if checkpoint exists
                    if [ ! -f "$ckpt_path" ]; then
                        echo "Skipping: Checkpoint not found: $ckpt_path"
                        continue
                    fi
                    
                    echo "Launching evaluation: seed=$seed, max_demos=$max_demos, use_gt_ref=$use_gt_ref, share_noise=$share_noise, ckpt=$ckpt_type"
                    sbatch scripts_launch/job_eval_local_flow_2d.sh \
                        --ckpt-path "$ckpt_path" \
                        --num-envs $NUM_ENVS \
                        --max-steps $MAX_STEPS
                done
            done
        done
    done
done

echo "All evaluation jobs launched. Check logs_berzelius/ for output."


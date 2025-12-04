#!/bin/bash
# Launch training jobs for local flow 2D policy with different hyperparameters
# Supports training with or without ground truth reference frame

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Define hyperparameter sweeps
SEEDS=(1234)
LEARNING_RATES=(1e-4)
BATCH_SIZES=(256)
MAX_DEMOS=(200)
USE_GT_REFERENCE_OPTIONS=(true false)  # Train with and without GT reference frame
SHARE_NOISE_OPTIONS=(true false)  # Train with and without shared noise

# Launch jobs for each combination
for seed in "${SEEDS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for max_demos in "${MAX_DEMOS[@]}"; do
                for use_gt_ref in "${USE_GT_REFERENCE_OPTIONS[@]}"; do
                    for share_noise in "${SHARE_NOISE_OPTIONS[@]}"; do
                        echo "Launching job: seed=$seed, lr=$lr, batch_size=$bs, max_demos=$max_demos, use_gt_reference=$use_gt_ref, share_noise=$share_noise"
                        
                        # Build sbatch command
                        sbatch_args=(
                            "scripts_launch/job_training_local_flow_2d.sh"
                            "--seed" "$seed"
                            "--lr" "$lr"
                            "--batch-size" "$bs"
                            "--max-demos" "$max_demos"
                            "--wandb"
                        )
                        
                        # Add GT reference flag if needed
                        if [ "$use_gt_ref" = true ]; then
                            sbatch_args+=("--use-gt-reference")
                        fi
                        
                        # Add share noise flag if needed
                        if [ "$share_noise" = true ]; then
                            sbatch_args+=("--share-noise")
                        fi
                        
                        sbatch "${sbatch_args[@]}"
                    done
                done
            done
        done
    done
done

echo "All jobs launched. Check logs_berzelius/ for output."


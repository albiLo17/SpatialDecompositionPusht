#!/bin/bash
# Launch training jobs for local flow 2D policy with different hyperparameters
# Supports training with or without ground truth reference frame

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Define hyperparameter sweeps
SEEDS=(2222)  # 1234
LEARNING_RATES=(1e-4)
BATCH_SIZES=(256)
MAX_DEMOS=(20 60 100 140 180 200)
USE_GT_REFERENCE_OPTIONS=(true)  # Train with and without GT reference frame
SHARE_NOISE_OPTIONS=(false)  # Train with and without shared noise
PARTICLES_AGGREGATION_OPTIONS=(median knn)  # Particle aggregation method: median or knn
USE_FILM_OPTIONS=(false true)  # Use FiLM conditioning: false or true

# Launch jobs for each combination
for seed in "${SEEDS[@]}"; do
    for lr in "${LEARNING_RATES[@]}"; do
        for bs in "${BATCH_SIZES[@]}"; do
            for max_demos in "${MAX_DEMOS[@]}"; do
                for use_gt_ref in "${USE_GT_REFERENCE_OPTIONS[@]}"; do
                    for share_noise in "${SHARE_NOISE_OPTIONS[@]}"; do
                        for particles_agg in "${PARTICLES_AGGREGATION_OPTIONS[@]}"; do
                            for use_film in "${USE_FILM_OPTIONS[@]}"; do
                                echo "Launching job: seed=$seed, lr=$lr, batch_size=$bs, max_demos=$max_demos, use_gt_reference=$use_gt_ref, share_noise=$share_noise, particles_agg=$particles_agg, use_film=$use_film"
                                
                                # Build sbatch command
                                sbatch_args=(
                                    "scripts_launch/job_training_local_flow_2d.sh"
                                    "--seed" "$seed"
                                    "--lr" "$lr"
                                    "--batch-size" "$bs"
                                    "--max-demos" "$max_demos"
                                    "--wandb"
                                    "--use-position-decoder"
                                    "--position-loss-coeff" "1.0"
                                    "--contact-threshold" "0.1"
                                    "--position-decoder-particles-aggregation" "$particles_agg"
                                )
                                
                                # Add GT reference flag if needed
                                if [ "$use_gt_ref" = true ]; then
                                    sbatch_args+=("--use-gt-reference")
                                fi
                                
                                # Add share noise flag if needed
                                if [ "$share_noise" = true ]; then
                                    sbatch_args+=("--share-noise")
                                fi
                                
                                # Add FiLM conditioning flag if needed
                                if [ "$use_film" = true ]; then
                                    sbatch_args+=("--use-film-conditioning")
                                fi
                                
                                sbatch "${sbatch_args[@]}"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All jobs launched. Check logs_berzelius/ for output."


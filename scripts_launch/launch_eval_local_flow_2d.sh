#!/bin/bash
# Launch evaluation jobs for different trained local flow 2D models

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Base directory for experiments
BASE_DIR="log/dp"

# Define which experiments to evaluate
# You can customize these arrays to match your training experiments
SEEDS=(2222) # 1234
MAX_DEMOS=(20 60 100 140 180 200)
USE_GT_REFERENCE_OPTIONS=(true)
SHARE_NOISE_OPTIONS=(false)
PARTICLES_AGGREGATION_OPTIONS=(knn)  # Particle aggregation method: median or knn
USE_FILM_OPTIONS=(true)  # Use FiLM conditioning: false or true
DISABLE_REFERENCE_CONDITIONING_OPTIONS=(true) # true)  # Disable reference conditioning (baseline ablation): false or true

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
                for particles_agg in "${PARTICLES_AGGREGATION_OPTIONS[@]}"; do
                    for use_film in "${USE_FILM_OPTIONS[@]}"; do
                        for disable_ref_cond in "${DISABLE_REFERENCE_CONDITIONING_OPTIONS[@]}"; do
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
                                if [ "$use_film" = true ]; then
                                    film_str="film"
                                else
                                    film_str="concat"
                                fi
                                # Base experiment name
                                exp_name_base="sd-pusht-local-flow-2d_V3-demos-${max_demos}-seed${seed}-${noise_str}-${ref_str}-${film_str}-${particles_agg}"
                                
                                # Try to find checkpoint - check both with and without -no-ref-cond suffix
                                # First try the expected name based on disable_ref_cond flag
                                if [ "$disable_ref_cond" = true ]; then
                                    exp_name_try="${exp_name_base}-no-ref-cond"
                                else
                                    exp_name_try="${exp_name_base}"
                                fi
                                
                                ckpt_path="${BASE_DIR}/${exp_name_try}/checkpoints/${ckpt_type}"
                                
                                # If not found and we're evaluating with ablation, try without ablation suffix
                                if [ ! -f "$ckpt_path" ] && [ "$disable_ref_cond" = true ]; then
                                    ckpt_path="${BASE_DIR}/${exp_name_base}/checkpoints/${ckpt_type}"
                                fi
                                
                                # Check if checkpoint exists
                                if [ ! -f "$ckpt_path" ]; then
                                    echo "Skipping: Checkpoint not found: $ckpt_path"
                                    continue
                                fi
                                
                                echo "Launching evaluation: seed=$seed, max_demos=$max_demos, use_gt_ref=$use_gt_ref, share_noise=$share_noise, particles_agg=$particles_agg, use_film=$use_film, disable_ref_cond=$disable_ref_cond, ckpt=$ckpt_type"
                                
                                # Build sbatch command
                                eval_args=(
                                    "scripts_launch/job_eval_local_flow_2d.sh"
                                    "--ckpt-path" "$ckpt_path"
                                    "--num-envs" "$NUM_ENVS"
                                    "--max-steps" "$MAX_STEPS"
                                    "--position-decoder-particles-aggregation" "$particles_agg"
                                    "--use-film-conditioning" "$use_film"
                                )
                                
                                # Add disable reference conditioning flag if needed
                                if [ "$disable_ref_cond" = true ]; then
                                    eval_args+=("--disable-reference-conditioning")
                                fi
                                
                                sbatch "${eval_args[@]}"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "All evaluation jobs launched. Check logs_berzelius/ for output."


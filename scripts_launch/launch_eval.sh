#!/bin/bash
# Launch evaluation jobs for different trained models

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Base directory for experiments
BASE_DIR="log/dp"

# Define which experiments to evaluate
# You can customize these arrays to match your training experiments
MODELS=("diffusion" "flowmatching")
SEEDS=(1234)
MAX_DEMOS=(20 60 100 140 180 200)

# Checkpoint types to evaluate (final model and/or best model)
CHECKPOINT_TYPES=("best_ema_model.pt")

# Evaluation parameters (can be overridden per job)
NUM_ENVS=128
MAX_STEPS=300

# Launch jobs for each combination
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for max_demos in "${MAX_DEMOS[@]}"; do
            for ckpt_type in "${CHECKPOINT_TYPES[@]}"; do
                # Construct experiment name
                if [ "$model" == "flowmatching" ]; then
                    model_type="fm"
                else
                    model_type="diffusion"
                fi
                exp_name="sd-pusht-${model_type}-demos-${max_demos}-seed${seed}"
                ckpt_path="${BASE_DIR}/${exp_name}/checkpoints/${ckpt_type}"
                
                # Check if checkpoint exists
                if [ ! -f "$ckpt_path" ]; then
                    echo "Skipping: Checkpoint not found: $ckpt_path"
                    continue
                fi
                
                echo "Launching evaluation: model=$model, seed=$seed, max_demos=$max_demos, ckpt=$ckpt_type"
                sbatch scripts_launch/job_eval.sh \
                    --ckpt-path "$ckpt_path" \
                    --num-envs $NUM_ENVS \
                    --max-steps $MAX_STEPS
            done
        done
    done
done

echo "All evaluation jobs launched. Check logs_berzelius/ for output."


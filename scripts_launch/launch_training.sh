#!/bin/bash
# Launch training jobs with different hyperparameters

# Create logs directory if it doesn't exist
mkdir -p logs_berzelius

# Define hyperparameter sweeps
MODELS=("diffusion" "flowmatching")
SEEDS=(1234)
LEARNING_RATES=(3e-4)
BATCH_SIZES=(256)
MAX_DEMOS=(20 60 100 140 180 200)

# Launch jobs for each combination
for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                for max_demos in "${MAX_DEMOS[@]}"; do
                    echo "Launching job: model=$model, seed=$seed, lr=$lr, batch_size=$bs, max_demos=$max_demos"
                    sbatch scripts_launch/job_training.sh \
                        --model $model \
                        --seed $seed \
                        --lr $lr \
                        --batch-size $bs \
                        --max-demos $max_demos \
                        --wandb
                done
            done
        done
    done
done

echo "All jobs launched. Check logs_berzelius/ for output."

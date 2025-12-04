#!/bin/bash
# SLURM batch job script for training diffusion/flow matching models

#SBATCH -A Berzelius-2025-278
#SBATCH --job-name=sd_pusht_train
#SBATCH --gpus=1                           # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH -o logs_berzelius/slurm-%j.out     # Job output
#SBATCH -e logs_berzelius/slurm-%j.err     # Job errors
#SBATCH --reservation=safe

# Parse command line arguments
USE_FLOW_MATCHING=false
SEED=0
LR=1e-4
BATCH_SIZE=256
MAX_DEMOS=1000
WANDB=false
WANDB_PROJECT="sd-pusht"
EPOCHS=1000
EVAL_EVERY=10
NUM_DIFFUSION_ITERS=100
FM_TIMESTEPS=100
SIGMA=0.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [ "$2" == "flowmatching" ]; then
                USE_FLOW_MATCHING=true
            fi
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-demos)
            MAX_DEMOS="$2"
            shift 2
            ;;
        --wandb)
            WANDB=true
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --eval-every)
            EVAL_EVERY="$2"
            shift 2
            ;;
        --num-diffusion-iters)
            NUM_DIFFUSION_ITERS="$2"
            shift 2
            ;;
        --fm-timesteps)
            FM_TIMESTEPS="$2"
            shift 2
            ;;
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Print job information
echo "=========================================="
echo "Starting SLURM job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo "Model: $([ "$USE_FLOW_MATCHING" = true ] && echo "Flow Matching" || echo "Diffusion")"
echo "Seed: $SEED"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Demos: $MAX_DEMOS"
echo "Epochs: $EPOCHS"
echo "WandB: $WANDB"
echo "=========================================="

# Change to project directory
cd /proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht

# Source environment setup
if [ -f setup_env.sh ]; then
    source setup_env.sh
else
    echo "Warning: setup_env.sh not found, skipping environment setup"
fi

# Build command arguments
CMD_ARGS=(
    "SD_pusht/scripts/train.py"
    "--seed" "$SEED"
    "--lr" "$LR"
    "--batch-size" "$BATCH_SIZE"
    "--max-demos" "$MAX_DEMOS"
    "--epochs" "$EPOCHS"
    "--eval-every" "$EVAL_EVERY"
    "--num-diffusion-iters" "$NUM_DIFFUSION_ITERS"
    "--fm-timesteps" "$FM_TIMESTEPS"
    "--sigma" "$SIGMA"
)

# Add model-specific flag
if [ "$USE_FLOW_MATCHING" = true ]; then
    CMD_ARGS+=("--use-flow-matching")
fi

# Add wandb flag if enabled
if [ "$WANDB" = true ]; then
    CMD_ARGS+=("--wandb" "--wandb-project" "$WANDB_PROJECT")
fi

# Run training
echo "Running: python ${CMD_ARGS[*]}"
echo ""

srun python "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE

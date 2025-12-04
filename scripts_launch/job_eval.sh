#!/bin/bash
# SLURM batch job script for evaluating diffusion/flow matching models

#SBATCH -A Berzelius-2025-278
#SBATCH --job-name=sd_pusht_eval
#SBATCH --gpus=1                           # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=4:00:00                     # Evaluation is faster than training
#SBATCH -o logs_berzelius/slurm-eval-%j.out     # Job output
#SBATCH -e logs_berzelius/slurm-eval-%j.err     # Job errors
#SBATCH --reservation=safe

# Parse command line arguments
CKPT_PATH=""
NUM_ENVS=64
MAX_STEPS=300
DEVICE=""
OUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --num-envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --out-path)
            OUT_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Validate checkpoint path
if [ -z "$CKPT_PATH" ]; then
    echo "Error: --ckpt-path is required"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

# Print job information
echo "=========================================="
echo "Starting SLURM evaluation job $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo "Checkpoint: $CKPT_PATH"
echo "Number of Environments: $NUM_ENVS"
echo "Max Steps: $MAX_STEPS"
if [ -n "$DEVICE" ]; then
    echo "Device: $DEVICE"
fi
if [ -n "$OUT_PATH" ]; then
    echo "Output Path: $OUT_PATH"
fi
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
    "SD_pusht/scripts/eval.py"
    "--ckpt-path" "$CKPT_PATH"
    "--num-envs" "$NUM_ENVS"
    "--max-steps" "$MAX_STEPS"
)

# Add optional arguments
if [ -n "$DEVICE" ]; then
    CMD_ARGS+=("--device" "$DEVICE")
fi

if [ -n "$OUT_PATH" ]; then
    CMD_ARGS+=("--out-path" "$OUT_PATH")
fi

# Run evaluation
echo "Running: python ${CMD_ARGS[*]}"
echo ""

srun python "${CMD_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Evaluation finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE


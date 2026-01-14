#!/bin/bash
# SLURM batch job script for training local flow 2D policy

#SBATCH -A Berzelius-2025-278
#SBATCH --job-name=sd_pusht_local_flow_2d
#SBATCH --gpus=1                           # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH -o logs_berzelius/slurm-%j.out     # Job output
#SBATCH -e logs_berzelius/slurm-%j.err     # Job errors
#SBATCH --reservation=safe

# Parse command line arguments
SEED=0
LR=1e-4
BATCH_SIZE=256
MAX_DEMOS=1000
WANDB=false
WANDB_PROJECT="sd-pusht-local-flow-2d"
EPOCHS=1000
EVAL_EVERY=10
FM_TIMESTEPS=100
SIGMA=0.0
USE_GT_REFERENCE=false
DATASET_PATH="datasets/pusht_cchi_v7_replay.zarr.zip"
OUTPUT_DIR="log/dp"

# LocalFlowPolicy2D specific arguments
USE_POSITION_DECODER=true
POSITION_DECODER_DOWN_DIMS=(256)
POSITION_DECODER_N_GROUPS=4
POSITION_DECODER_FM_TIMESTEPS=8
POSITION_LOSS_COEFF=1.0
SHARE_NOISE=false
SHARED_NOISE_BASE="action"

# Dataset segmentation arguments
CONTACT_THRESHOLD=0.1
MIN_SEGMENT_LENGTH=5

# Model hyperparameters
PRED_HORIZON=16
OBS_HORIZON=2
ACTION_HORIZON=8
OBS_DIM=5
ACTION_DIM=2

while [[ $# -gt 0 ]]; do
    case $1 in
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
        --fm-timesteps)
            FM_TIMESTEPS="$2"
            shift 2
            ;;
        --sigma)
            SIGMA="$2"
            shift 2
            ;;
        --use-gt-reference)
            USE_GT_REFERENCE=true
            shift
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use-position-decoder)
            USE_POSITION_DECODER=true
            shift
            ;;
        --no-position-decoder)
            USE_POSITION_DECODER=false
            shift
            ;;
        --position-loss-coeff)
            POSITION_LOSS_COEFF="$2"
            shift 2
            ;;
        --share-noise)
            SHARE_NOISE=true
            shift
            ;;
        --shared-noise-base)
            SHARED_NOISE_BASE="$2"
            shift 2
            ;;
        --contact-threshold)
            CONTACT_THRESHOLD="$2"
            shift 2
            ;;
        --min-segment-length)
            MIN_SEGMENT_LENGTH="$2"
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
echo "Model: LocalFlowPolicy2D"
echo "Seed: $SEED"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "Max Demos: $MAX_DEMOS"
echo "Epochs: $EPOCHS"
echo "WandB: $WANDB"
echo "Use GT Reference: $USE_GT_REFERENCE"
echo "Share Noise: $SHARE_NOISE"
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
    "SD_pusht/scripts/train_local_flow_2d.py"
    "--seed" "$SEED"
    "--lr" "$LR"
    "--batch-size" "$BATCH_SIZE"
    "--max-demos" "$MAX_DEMOS"
    "--epochs" "$EPOCHS"
    "--eval-every" "$EVAL_EVERY"
    "--fm-timesteps" "$FM_TIMESTEPS"
    "--sigma" "$SIGMA"
    "--dataset-path" "$DATASET_PATH"
    "--output-dir" "$OUTPUT_DIR"
    "--contact-threshold" "$CONTACT_THRESHOLD"
    "--min-segment-length" "$MIN_SEGMENT_LENGTH"
    "--position-loss-coeff" "$POSITION_LOSS_COEFF"
    "--shared-noise-base" "$SHARED_NOISE_BASE"
)

# Add conditional flags
if [ "$USE_POSITION_DECODER" = true ]; then
    CMD_ARGS+=("--use-position-decoder")
fi

if [ "$SHARE_NOISE" = true ]; then
    CMD_ARGS+=("--share-noise")
fi

if [ "$USE_GT_REFERENCE" = true ]; then
    CMD_ARGS+=("--use-gt-reference-for-local-policy")
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


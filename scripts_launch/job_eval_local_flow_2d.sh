#!/bin/bash
# SLURM batch job script for evaluating local flow 2D models

#SBATCH -A Berzelius-2025-278
#SBATCH --job-name=sd_pusht_eval_local_flow_2d
#SBATCH --gpus=1                           # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --time=4:00:00                     # Evaluation is faster than training
#SBATCH -o logs_berzelius/slurm-eval-local-flow-2d-%j.out     # Job output
#SBATCH -e logs_berzelius/slurm-eval-local-flow-2d-%j.err     # Job errors
#SBATCH --reservation=safe

# Parse command line arguments
CKPT_PATH=""
NUM_ENVS=64
MAX_STEPS=300
DEVICE=""
OUT_PATH=""
POSITION_DECODER_PARTICLES_AGGREGATION="median"
USE_FILM_CONDITIONING="false"
FILM_HIDDEN_DIM=64
FILM_PREDICT_SCALE="true"
DISABLE_REFERENCE_CONDITIONING="false"

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
        --position-decoder-particles-aggregation)
            POSITION_DECODER_PARTICLES_AGGREGATION="$2"
            shift 2
            ;;
        --use-film-conditioning)
            USE_FILM_CONDITIONING="$2"
            shift 2
            ;;
        --film-hidden-dim)
            FILM_HIDDEN_DIM="$2"
            shift 2
            ;;
        --film-predict-scale)
            FILM_PREDICT_SCALE="$2"
            shift 2
            ;;
        --disable-reference-conditioning)
            DISABLE_REFERENCE_CONDITIONING="true"
            shift
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
echo "Particles Aggregation: $POSITION_DECODER_PARTICLES_AGGREGATION"
echo "Use FiLM Conditioning: $USE_FILM_CONDITIONING"
echo "Disable Reference Conditioning: $DISABLE_REFERENCE_CONDITIONING"
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
    "SD_pusht/scripts/eval_local_flow_2d.py"
    "--ckpt-path" "$CKPT_PATH"
    "--num-envs" "$NUM_ENVS"
    "--max-steps" "$MAX_STEPS"
    "--position-decoder-particles-aggregation" "$POSITION_DECODER_PARTICLES_AGGREGATION"
    "--use-film-conditioning" "$USE_FILM_CONDITIONING"
)

# Add optional arguments
if [ -n "$DEVICE" ]; then
    CMD_ARGS+=("--device" "$DEVICE")
fi

if [ -n "$OUT_PATH" ]; then
    CMD_ARGS+=("--out-path" "$OUT_PATH")
fi

# Add FiLM arguments if provided
CMD_ARGS+=("--film-hidden-dim" "$FILM_HIDDEN_DIM")
CMD_ARGS+=("--film-predict-scale" "$FILM_PREDICT_SCALE")

# Add conditional flags
if [ "$DISABLE_REFERENCE_CONDITIONING" = "true" ]; then
    CMD_ARGS+=("--disable-reference-conditioning")
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


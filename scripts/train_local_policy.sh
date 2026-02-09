#!/bin/bash
# SLURM batch job script — run all scenarios as an array

#SBATCH -A Berzelius-2025-278
#SBATCH --job-name=lp_train
#SBATCH --gpus=1                           # 1 GPU per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH -o logs/slurm-%A_%a.out                 # no logs/ folder needed
#SBATCH -e logs/slurm-%A_%a.err
#SBATCH --reservation=safe

# add printing of the job number and node
echo "Starting SLURM job $SLURM_JOB_ID, node $(hostname)"

cd /proj/rep-learning-robotics/users/x_alblo/SpatialDecompositionPusht               

module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0

conda activate SD_pusht

export PYTHONUNBUFFERED=1


# srun python spatialdecomposition/scripts/training/train_local_policy.py \
#     --config.model.policy-class MLPPolicy \
#     --config.subtasks_embedding_size 16 \
#     --config.dataset Toy \
#     --config.dataset_path datasets/pusht_toy_dataset_segmented_normalized.npz \
#     --config.model.no-use-time \
#     --config.model.action_dim 2 \
#     --config.data_loader_workers 0 \
#     --config.epochs 2000 \
#     --config.wandb_log \
#     --config.wandb_entity albilo



srun python spatialdecomposition/scripts/training/train_local_policy.py \
    --config.model.policy-class MLPPolicy \
    --config.subtasks_embedding_size 32 \
    --config.dataset Toy \
    --config.dataset_path datasets/pusht_toy_dataset_segmented_normalized.npz \
    --config.model.no-use-time \
    --config.model.action_dim 2 \
    --config.data_loader_workers 0 \
    --config.epochs 2000 \
    --config.wandb_log \
    --config.wandb_entity albilo

echo "[$(date)] Finished training"
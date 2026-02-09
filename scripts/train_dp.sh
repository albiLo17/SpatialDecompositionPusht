#!/bin/bash
# SLURM batch job script — run all scenarios as an array

#SBATCH -A=Berzelius-2025-278
#SBATCH --job-name=dp_train
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

NUM_DEMO=$1

module load Miniforge3/24.7.1-2-hpc1-bdist
module load buildenv-gcccuda/12.1.1-gcc12.3.0

conda activate SD_pusht

export PYTHONUNBUFFERED=1

echo "[$(date)] Starting training with ${NUM_DEMO} demonstrations"
srun python SD_pusht/train_diffusion.py --max_demos "${NUM_DEMO}"
echo "[$(date)] Finished training with ${NUM_DEMO} demonstrations"
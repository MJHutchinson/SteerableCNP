#!/bin/bash

# Usage: `sbatch`

#SBATCH --job-name=EquivCNP_test
#SBATCH --partition=ziz-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=14-00:00:00
#SBATCH --mem=10G
#SBATCH --ntasks=1
#SBATCH --output=/data/ziz/not-backed-up/mhutchin/slurm_logs/equiv_cnp/slurm-GP_test-%A_%a.out

source /data/ziz/not-backed-up/mhutchin/EquivCNP/.venv/bin/activate

python train_gp_data.py
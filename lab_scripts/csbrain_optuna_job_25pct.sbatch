#!/bin/bash
#SBATCH --job-name=csbrain_optuna
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nourmaj/links/scratch/csbrain_optuna_logs/25pct/%x_%j.out

source /lustre10/scratch/nourmaj/envs/cbramod-gpu/bin/activate
export HOME=/home/nourmaj/links/scratch/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/home/nourmaj/links/scratch/numba_cache
mkdir -p $NUMBA_CACHE_DIR

mkdir -p /home/nourmaj/links/scratch/csbrain_optuna_logs/25pct
mkdir -p /home/nourmaj/links/scratch/csbrain_optuna_results

python /home/nourmaj/links/scratch/csbrain/csbrain_optuna_search.py \
    --dataset "$DATASET" \
    --arm "$ARM" \
    --n_trials "$N_TRIALS" \
    --fraction 25

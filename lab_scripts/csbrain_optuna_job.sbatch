#!/bin/bash
#SBATCH --job-name=csbrain_optuna
# (overridden per-submission by whatever --job-name you pass at sbatch time,
# same reasoning as cbramod_optuna_job.sbatch: #SBATCH directives don't
# expand --export'd shell vars, only sbatch's own CLI flags do.)
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/nourmaj/links/scratch/csbrain_optuna_logs/%x_%j.out

source /lustre10/scratch/nourmaj/envs/cbramod-gpu/bin/activate
export HOME=/home/nourmaj/links/scratch/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/home/nourmaj/links/scratch/numba_cache
mkdir -p $NUMBA_CACHE_DIR

mkdir -p /home/nourmaj/links/scratch/csbrain_optuna_logs
mkdir -p /home/nourmaj/links/scratch/csbrain_optuna_results

# DATASET / ARM / N_TRIALS passed in via sbatch --export=DATASET=...,ARM=...,N_TRIALS=...
# Study state lives in a per-run sqlite db (load_if_exists=True), so a
# timed-out/re-submitted job just picks up where it left off.
python /home/nourmaj/links/scratch/csbrain/csbrain_optuna_search.py \
    --dataset "$DATASET" \
    --arm "$ARM" \
    --n_trials "$N_TRIALS"

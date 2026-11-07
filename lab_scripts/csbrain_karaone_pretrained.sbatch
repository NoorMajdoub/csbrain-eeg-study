#!/bin/bash
#SBATCH --job-name=csbrain_karaone_pt
#SBATCH --account=def-amiilab_gpu
#SBATCH --partition=gpubase_bygpu_b2
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/nourmaj/links/scratch/csbrain_karaone_pretrained_%j.out

source /lustre10/scratch/nourmaj/envs/cbramod-gpu/bin/activate
export HOME=/home/nourmaj/links/scratch/fake_home
mkdir -p $HOME
export NUMBA_DISABLE_JIT=1
export NUMBA_CACHE_DIR=/home/nourmaj/links/scratch/numba_cache
mkdir -p $NUMBA_CACHE_DIR

cd /home/nourmaj/links/scratch/csbrain/CSBrain

python finetune_main.py --downstream_dataset KaraOne --datasets_dir /home/nourmaj/links/scratch/karaone_processed --num_of_classes 2 --model_dir /home/nourmaj/links/scratch/csbrain_pth_downtasks/karaone_pretrained --model CSBrain --dropout 0.1 --weight_decay 0.05 --lr 0.0001 --batch_size 32 --use_pretrained_weights --foundation_dir pth/CSBrain.pth

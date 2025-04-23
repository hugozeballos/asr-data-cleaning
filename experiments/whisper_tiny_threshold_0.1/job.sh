#!/bin/bash
#SBATCH --job-name=whisper_tiny_threshold_0.1
#SBATCH --output=/mnt/ialabnas/homes/hugoz/asr-data-cleaning/experiments/whisper_tiny_threshold_0.1/logs/whisper_tiny_threshold_0.1_%j.log
#SBATCH --error=/mnt/ialabnas/homes/hugoz/asr-data-cleaning/experiments/whisper_tiny_threshold_0.1/logs/whisper_tiny_threshold_0.1_%j.err
#SBATCH --partition=ialab-high
#SBATCH --gres=gpu:titan_rtx:2
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4

# Move to project root
cd ~/storage/asr-data-cleaning

# Load environment
source ~/.bashrc
conda activate whisper_env

# Hugging Face and PyTorch configs
export HF_HOME=Hugginsface_cache
export HF_DATASETS_CACHE=datasets_cache
export TMPDIR=tmp
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run training
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --rdzv_backend=static \
  --rdzv_endpoint=localhost:29501 \
  main.py --config_path=/mnt/ialabnas/homes/hugoz/asr-data-cleaning/experiments/whisper_tiny_threshold_0.1/experiment_config.json

# Post-run cleanup
rm -rf /home/hugoz/.cache/torch

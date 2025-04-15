#!/bin/bash
#SBATCH --job-name=whysper_filtering_distrib_ventress
#SBATCH --output=whysper_filtering_distrib_%j.log
#SBATCH --error=whysper_filtering_distrib_%j.err
#SBATCH --partition=ialab-high
#SBATCH --gres=gpu:1080_ti:2
#SBATCH --mem=32G
#SBATCH --time=2-20:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4

# Muestra el directorio actual, el hostname y la fecha
pwd; hostname; date

#echo "Limpiando cache GPU..."
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Ir al directorio donde se encuentra el código
cd /home/hugoz/storage/whisper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
huggingface-cli login --token REDACTED_HF_TOKEN
export CUDA_LAUNCH_BLOCKING=1
echo "Ejecutando iterative_filter.py en modo test..."

export HF_HOME=Hugginsface_cache
export HF_DATASETS_CACHE=datasets_cache
export TMPDIR=tmp
#export ACCELERATE_CONFIG_DIR=/home/hugoz/storage/whisper/accelerate_config
#export ACCELERATE_CONFIG_FILE=~/.cache/huggingface/accelerate/default_config.yaml

echo "Usando configuración de Accelerate en: $ACCELERATE_CONFIG_FILE"

#export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
#export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
#export WORLD_SIZE=$SLURM_NTASKS
#export RANK=$SLURM_PROCID
#export LOCAL_RANK=$SLURM_LOCALID
#export NODE_RANK=$SLURM_NODEID

echo "Ejecutando entrenamiento con Accelerate..."
#SBATCH --gpus=2  # Usa solo 2 GPUs
#export CUDA_VISIBLE_DEVICES=0,1,5
#accelerate launch --main_process_port 29501 main.py

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=static --rdzv_endpoint=localhost:29501 main.py
#torchrun --nnodes=1 --nproc_per_node=2 main.py
#accelerate launch main.py

#accelerate env
#accelerate launch \
#    --config_file /home/hugoz/storage/whisper/accelerate_config/default_config.yaml \
#    --multi_gpu \
#    --main_process_port 0 \
#    --num_processes $WORLD_SIZE \
#    --num_machines 1 \
#    --mixed_precision fp16 \
#    --main_process_port=$MASTER_PORT \
#    fine_tuning_ventress.py

# En este ejemplo, se asume que en el bloque __main__ del script se activa el modo test
echo "Limpiando caches locales..."
#rm -rf /home/hugoz/.cache/huggingface
rm -rf /home/hugoz/.cache/torch

echo "Trabajo finalizado con job $SLURM_JOBID"
date

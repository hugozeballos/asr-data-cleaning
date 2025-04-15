#!/bin/bash
#SBATCH --job-name=whysper_filtering_distrib_ashoka
#SBATCH --output=whysper_filtering_distrib_%j.log
#SBATCH --error=whysper_filtering_distrib_%j.err
#SBATCH --partition=ialab-low
#SBATCH --gres=gpu:titan_x:3
#SBATCH --mem=32G
#SBATCH --time=2-20:50:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1

# Muestra el directorio actual, el hostname y la fecha
pwd; hostname; date

echo "Limpiando cache GPU..."
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
# En este ejemplo, se asume que en el bloque __main__ del script se activa el modo test
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_backend=static --rdzv_endpoint=localhost:29501 main.py

echo "Limpiando caches locales..."
rm -rf /home/hugoz/.cache/huggingface
rm -rf /home/hugoz/.cache/torch

echo "Trabajo finalizado con job $SLURM_JOBID"
date

#!/bin/bash
#SBATCH --job-name=whysper_filtering_lora
#SBATCH --output=whysper_filtering_lora_%j.log
#SBATCH --error=whysper_filtering_lora_%j.err
#SBATCH --partition=ialab-high
#SBATCH --gres=gpu:titan_rtx:1
#SBATCH --mem=32G
#SBATCH --time=0-50:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Muestra el directorio actual, el hostname y la fecha
pwd; hostname; date

echo "Limpiando cache GPU..."
python -c "import torch; torch.cuda.empty_cache(); import gc; gc.collect()"

# Ir al directorio donde se encuentra el código
cd /home/hugoz/storage/whisper
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
huggingface-cli login --token REDACTED_HF_TOKEN
echo "Ejecutando iterative_filter.py en modo test..."
# En este ejemplo, se asume que en el bloque __main__ del script se activa el modo test
python iterative_filter_lora.py

echo "Limpiando caches locales..."
rm -rf /home/hugoz/.cache/huggingface
rm -rf /home/hugoz/.cache/torch

echo "Trabajo finalizado con job $SLURM_JOBID"
date

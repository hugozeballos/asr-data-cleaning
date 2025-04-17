#!/bin/bash

# Path to the experiment configuration
CONFIG="experiment_config.json"

# Helper to extract values from JSON
CFG_PY="import json; c=json.load(open('$CONFIG')); print(c.get('{key}', ''))"

# Read values from the config
RUN_NAME=$(python3 -c "${CFG_PY//\{key\}/run_name}")
GPUS=$(python3 -c "${CFG_PY//\{key\}/gpus}")
GPU_TYPE=$(python3 -c "${CFG_PY//\{key\}/gpu_type}")
PARTITION=$(python3 -c "${CFG_PY//\{key\}/partition}")
NTASKS=$(python3 -c "${CFG_PY//\{key\}/ntasks}")
CPUS=$(python3 -c "${CFG_PY//\{key\}/cpus_per_task}")

# Validate required fields
if [[ -z "$RUN_NAME" || -z "$GPUS" || -z "$GPU_TYPE" || -z "$PARTITION" || -z "$NTASKS" || -z "$CPUS" ]]; then
  echo "❌ Error: Some fields are missing in $CONFIG. Please check."
  exit 1
fi

# Prepare experiment directory
EXP_DIR="experiments/${RUN_NAME}"
mkdir -p "$EXP_DIR/logs"

# Backup the experiment configuration
cp "$CONFIG" "$EXP_DIR/experiment_config.json"

# Create the Slurm job script dynamically
cat <<EOF > "$EXP_DIR/job.sh"
#!/bin/bash
#SBATCH --job-name=$RUN_NAME
#SBATCH --output=$EXP_DIR/logs/${RUN_NAME}_%j.log
#SBATCH --error=$EXP_DIR/logs/${RUN_NAME}_%j.err
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:${GPU_TYPE}:${GPUS}
#SBATCH --mem=32G
#SBATCH --time=2-20:00:00
#SBATCH --ntasks=$NTASKS
#SBATCH --cpus-per-task=$CPUS

# Move to the project directory
cd /home/hugoz/storage/asr-data-cleaning

# Load environment
source ~/.bashrc
conda activate whisper_env

echo "🚀 Running $RUN_NAME..."

# Hugging Face cache and memory configs
export HF_HOME=Hugginsface_cache
export HF_DATASETS_CACHE=datasets_cache
export TMPDIR=tmp
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Launch training
torchrun \\
  --nnodes=1 \\
  --nproc_per_node=$GPUS \\
  --rdzv_backend=static \\
  --rdzv_endpoint=localhost:29501 \\
  main.py

# Clean up after job
rm -rf /home/hugoz/.cache/torch
echo "🧹 Cleaning up all checkpoints..."

rm -rf "$EXP_DIR/checkpoints"

echo "✅ All checkpoints deleted."

echo "✅ Job finished for $RUN_NAME"
EOF

# Submit the job to SLURM
chmod +x "$EXP_DIR/job.sh"
sbatch "$EXP_DIR/job.sh"

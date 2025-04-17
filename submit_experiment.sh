#!/bin/bash

# 🚀 Submit a new Whisper cleaning experiment to SLURM

# 1️⃣ Path to the experiment config (must exist in experiments/<run_name>/experiment_config.json)
CONFIG="experiment_config.json"

# 2️⃣ Helper: extract values from config using python
CFG_PY="import json; c=json.load(open('$CONFIG')); print(c.get('{key}', ''))"

# 3️⃣ Read required fields
RUN_NAME=$(python3 -c "${CFG_PY//\{key\}/run_name}")
GPUS=$(python3 -c "${CFG_PY//\{key\}/gpus}")
GPU_TYPE=$(python3 -c "${CFG_PY//\{key\}/gpu_type}")
PARTITION=$(python3 -c "${CFG_PY//\{key\}/partition}")
NTASKS=$(python3 -c "${CFG_PY//\{key\}/ntasks}")
CPUS=$(python3 -c "${CFG_PY//\{key\}/cpus_per_task}")

# 4️⃣ Validate fields
if [[ -z "$RUN_NAME" || -z "$GPUS" || -z "$GPU_TYPE" || -z "$PARTITION" || -z "$NTASKS" || -z "$CPUS" ]]; then
  echo "❌ Error: Some required fields are missing in $CONFIG. Please check."
  exit 1
fi

# 5️⃣ Set experiment folder
EXP_DIR="/home/hugoz/storage/asr-data-cleaning/experiments/$RUN_NAME"

# 6️⃣ Create logs folder if needed
mkdir -p "$EXP_DIR/logs"

# 7️⃣ Generate SLURM job script dynamically
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

# 1️⃣ Move to the project directory
cd /home/hugoz/storage/asr-data-cleaning

# 2️⃣ Load environment
source ~/.bashrc
conda activate whisper_env

# 3️⃣ Environment variables for caching and CUDA
export HF_HOME=Hugginsface_cache
export HF_DATASETS_CACHE=datasets_cache
export TMPDIR=tmp
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 4️⃣ Run main.py with proper config
echo "🚀 Launching training with config experiments/$RUN_NAME/experiment_config.json ..."
torchrun \\
  --nnodes=1 \\
  --nproc_per_node=$GPUS \\
  --rdzv_backend=static \\
  --rdzv_endpoint=localhost:29501 \\
  main.py --config_path=experiments/$RUN_NAME/experiment_config.json

# 5️⃣ Clean up heavy cache after run
echo "🧹 Cleaning up torch cache..."
rm -rf /home/hugoz/.cache/torch

# 6️⃣ Clean up checkpoints if they exist
echo "🧹 Cleaning up checkpoints..."
rm -rf "$EXP_DIR/checkpoints"

echo "✅ Experiment $RUN_NAME finished."
EOF

# 8️⃣ Submit to Slurm
chmod +x "$EXP_DIR/job.sh"
echo "💬 Everything is ready, submitting to SLURM..."
sbatch "$EXP_DIR/job.sh"

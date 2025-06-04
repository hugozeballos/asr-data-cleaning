# submit_comparacion.py

import argparse
import json
import os
import subprocess
import shutil

# 1️⃣ Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to comparacion_config.json")
args = parser.parse_args()


# 2️⃣ Load configuration
with open(args.config_path) as f:
    cfg = json.load(f)

# 3️⃣ Read important fields
model_name = cfg["model_name"]
gpus = cfg["gpus"]
gpu_type = cfg["gpu_type"]
partition = cfg["partition"]
ntasks = cfg["ntasks"]
cpus_per_task = cfg["cpus_per_task"]
memory = cfg.get("memory", "32G")
run_name = f"cmp_{model_name}"

# 4️⃣ Define paths
root_dir = os.getcwd()
exp_dir = os.path.join(root_dir, "comparacion", model_name)
logs_dir = os.path.join(exp_dir, "logs")
job_script = os.path.join(exp_dir, "job.sh")
copied_config_path = os.path.join(exp_dir, "comparacion_config.json")

# 5️⃣ Create experiment/logs directories
os.makedirs(logs_dir, exist_ok=True)

# 6️⃣ Copy config into experiment folder
shutil.copy(args.config_path, copied_config_path)

# 7️⃣ Write job.sh script
with open(job_script, "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --output={logs_dir}/{run_name}_%j.log
#SBATCH --error={logs_dir}/{run_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH --mem={memory}
#SBATCH --time=1-00:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}

cd ~/storage/asr-data-cleaning

source ~/.bashrc
conda activate whisper_env

export HF_HOME=Hugginsface_cache
export HF_DATASETS_CACHE=datasets_cache
export TMPDIR=tmp
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Ejecutar entrenamiento por thresholds
torchrun \
  --nnodes=1 \
  --nproc_per_node={gpus} \
  --rdzv_backend=static \
  --rdzv_endpoint=localhost:12342 \
  train_threshold.py --config_path {copied_config_path}

rm -rf /home/$USER/.cache/torch
""")

os.chmod(job_script, 0o755)

print(f"✅ Comparación preparada en: {exp_dir}")
print(f"✅ Job script creado: {job_script}")

# 8️⃣ Submit job
print("🚀 Enviando trabajo a SLURM...")
subprocess.run(["sbatch", job_script])
print("🎯 Trabajo enviado exitosamente!")
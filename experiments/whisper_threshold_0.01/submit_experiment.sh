import argparse
import json
import os
import subprocess

# 1️⃣ Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, required=True, help="Path to experiment_config.json")
args = parser.parse_args()

# 2️⃣ Load configuration
with open(args.config_path) as f:
    cfg = json.load(f)

# 3️⃣ Read important fields
run_name = cfg["run_name"]
gpus = cfg["gpus"]
gpu_type = cfg["gpu_type"]
partition = cfg["partition"]
ntasks = cfg["ntasks"]
cpus_per_task = cfg["cpus_per_task"]
memory = cfg.get("memory", "32G")  # Default to 32G

# 4️⃣ Define paths
exp_dir = os.path.join("experiments", run_name)
logs_dir = os.path.join(exp_dir, "logs")
job_script = os.path.join(exp_dir, "job.sh")

# 5️⃣ Create directories if missing
os.makedirs(logs_dir, exist_ok=True)

# 6️⃣ Write job.sh
with open(job_script, "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --job-name={run_name}
#SBATCH --output={logs_dir}/{run_name}_%j.log
#SBATCH --error={logs_dir}/{run_name}_%j.err
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{gpu_type}:{gpus}
#SBATCH --mem={memory}
#SBATCH --time=2-20:00:00
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}

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
srun torchrun \\
  --nnodes=1 \\
  --nproc_per_node={gpus} \\
  --rdzv_backend=static \\
  --rdzv_endpoint=localhost:29501 \\
  main.py --config_path={args.config_path}

# Post-run cleanup
rm -rf /home/hugoz/.cache/torch
""")

print(f"✅ Job script generated at: {job_script}")

# 7️⃣ Submit job to SLURM
print("🚀 Submitting job...")
subprocess.run(["sbatch", job_script])
print("🎯 Job submitted!")
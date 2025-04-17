import multiprocessing as mp
import os
import argparse
from utils import load_experiment_config
from train import train

if __name__ == "__main__":
    # 🔹 Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Path to the experiment_config.json")
    args = parser.parse_args()

    # 🔹 Load the experiment configuration
    cfg = load_experiment_config(args.config_path)

    # 🔹 Create output directories if they don't exist
    output_dir = os.path.join("experiments", cfg["run_name"])
    print(f"🚀 Running experiment in: {output_dir}")

    # 🔹 Set multiprocessing start method to "spawn" (required for torchrun compatibility)
    mp.set_start_method("spawn", force=True)

    # 🔹 Start the training process
    train(cfg)

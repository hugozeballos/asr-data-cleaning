# ASR Dataset Cleaning with Cross-Validation (Whisper)

A data-quality pipeline for Automatic Speech Recognition (ASR) training data, built as part of an AI-powered Rapa Nui language translator project developed during a contract at [CENIA](https://www.cenia.cl/) (Centro Nacional de Inteligencia Artificial, Chile's national AI research center).

The goal isn't to train the best possible model — it's to answer a data-quality question: **which filtering strategy produces the cleanest, most reliable dataset for ASR training?** Real-world speech datasets (especially for a low-resource language like Rapa Nui) contain mislabeled or low-quality samples that hurt model performance if left in. This pipeline identifies and evaluates candidate samples to remove, rather than committing to a single fixed cleaning rule.

## How it works

**Stage 1 — Cross-validation filtering** (`main.py` → `train.py`, config-driven via `experiment_config.json`)
The dataset is split into 10 folds. For each fold, a Whisper model (`openai/whisper-large-v3-turbo` in the current config, though the model is configurable) is fine-tuned on the other 9 folds and evaluated on the held-out fold. Per-sample Character Error Rate (CER) is computed on that held-out data, and results are checkpointed incrementally (fold-by-fold, resumable) to a `removed_ids_*.json`-style CER records file. This produces a CER score for every sample in the dataset without ever validating a sample against a model it helped train.

**Stage 2 — Threshold comparison** (`train_threshold.py` / `cv_compute_cer.py`, config-driven via `comparacion_config.json`)
Using the per-sample CER scores from Stage 1, the dataset is filtered at different CER thresholds, a fresh model is trained on each filtered version, and all variants are evaluated on a common held-out test set (5% of the data, excluded from filtering). Global and per-sample WER/CER for each threshold are logged, along with scatter and boxplot visualizations of CER distribution per threshold — making the cleaning-strategy comparison directly visible.

All experiments are tracked in [MLflow](https://mlflow.org/) (a remote tracking server) and submitted as jobs to a SLURM GPU cluster via `submit_experiment.py` and `submit_comparacion.py`, which generate and `sbatch` the job scripts.

The repository also includes `t2s_test.ipynb`, a separate exploratory notebook for testing Meta's MMS text-to-speech model on Rapa Nui audio generation — useful for spot-checking dataset audio/text pairs, but not part of the core CV-filtering pipeline. `fine_tuning_primero_funcional.py` is an earlier, standalone prototype fine-tuning script kept for reference; the CV pipeline (`main.py`/`train.py`) is the current approach.

## How it fits into the larger Rapa Nui translator project

This pipeline consumes a Rapa Nui speech dataset from the Hugging Face Hub (with pre-computed Whisper inferences and metrics already attached) and produces a cleaned dataset plus a comparison of cleaning strategies. The output — the filtered training data and the knowledge of which cleaning approach works best — is meant to feed into training/fine-tuning a production ASR model for the broader translator system, rather than being an ASR service itself.

## Tech stack

- **Python 3.10**, PyTorch 2.4 (CUDA 12.4/12.8)
- **Hugging Face**: `transformers` (Whisper model + `Seq2SeqTrainer`), `datasets` (Hub loading, folding, filtering), `evaluate` + `jiwer` (WER/CER metrics), `huggingface_hub`
- **MLflow** for experiment tracking (remote tracking server)
- **SLURM** for GPU cluster job submission (`sbatch`, multi-GPU via `torchrun`)
- **matplotlib** for CER distribution visualizations

## Running it

1. Create the conda environment: `conda env create -f environment.yml` (or `pip install -r requirements.txt` into an existing env — note the CUDA 12.4 wheel index).
2. Authenticate with Hugging Face (`huggingface-cli login`, or set the `HF_TOKEN` environment variable) — required to pull the training dataset and, if applicable, gated models.
3. Adjust `experiment_config.json` (model name, dataset name, batch size, GPU/partition settings) for a Stage 1 cross-validation run.
4. Submit to the cluster: `python submit_experiment.py --config_path experiment_config.json` (writes a SLURM job script under `experiments/<run_name>/` and calls `sbatch`), or run directly with `python main.py --config_path experiment_config.json` on a machine with a GPU.
5. Once Stage 1 CER records exist, configure `comparacion_config.json` (thresholds, filtered model name) and run Stage 2 similarly via `python submit_comparacion.py --config_path comparacion_config.json`.

Both stages are checkpointed/resumable and log to the MLflow tracking server configured in `train.py`/`cv_compute_cer.py`.

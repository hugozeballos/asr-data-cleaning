"""
train.py

Main training script for Whisper-based ASR model using 10-fold cross-validation.
For each fold, it filters samples based on CER and logs metrics to MLflow.

All configuration (model name, output paths, experiment name, etc.) is loaded
from a single JSON file: `experiment_config.json`.

Author: [CENIA]
"""
import shutil
import json
import os
import mlflow
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import Seq2SeqTrainer

from model import load_model
from dataset import (
    load_datasets,
    prepare_dataset,
    prepare_dataset_for_cross_validation,
    DataCollatorSpeechSeq2SeqWithPadding,
)
from config import get_training_args
from transformers.integrations import MLflowCallback
from transformers import Seq2SeqTrainer, EarlyStoppingCallback



from utils import (
    compute_metrics,
    compute_cer_per_fold
)

# --- is raank 0? for mlflow no repeat registerE ----------------------------------------------------------------
import torch.distributed as dist
from contextlib import nullcontext

def is_rank0() -> bool:
    """Devuelve True si no hay DDP o si el rank global es 0."""
    return (not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0)

def train(cfg):
    """
    Main training function. Performs cross-validation training with data filtering
    based on CER, and tracks results in MLflow.
    """
    # Load config
    training_args = get_training_args(cfg)
    #train_dataset, val_dataset = load_datasets()

    # Generate dataset folds (10-fold cross-validation)
    full_dataset, folds = prepare_dataset_for_cross_validation(cfg)

    # 1) Inicializar checkpoint de CER
    cer_file       = cfg["cer_records_file"]
    complete_sufx  = cfg["cer_complete_suffix"]
    cer_file_done  = cer_file.replace(".json", f"{complete_sufx}.json")
    cer_records    = []
    # Si ya existe el archivo final, salimos
    if os.path.exists(cer_file_done):
        print(f"✅ Experimento finalizado: {cer_file_done}")
        return

    # Si hay un checkpoint parcial, lo cargamos y determinamos desde qué fold arrancar
    if os.path.exists(cer_file):
        with open(cer_file) as f:
            cer_records = json.load(f)
        processed = {r["fold"] for r in cer_records}
        start_fold = max(processed) + 1 if processed else 0
        print(f"🔄 Reanudando desde fold {start_fold+1}")
    else:
        start_fold = 0
        print("🚀 Comenzando desde fold 1")
    

    # 🔄 Cross-validation loop (10 folds)
    for fold_idx in tqdm(range(start_fold, 10), desc="🔄 Cross-validation Progress"):
        # Load model and processor as specified in the config
        model, processor = load_model(cfg)
        print(f"\n📁 Fold {fold_idx+1}/10 - Using it as validation set")

        # Define train and validation indices
        val_indices = folds[fold_idx]
        train_indices = np.concatenate(
            [folds[i] for i in range(10) if i != fold_idx]
        )

        # Subset the original dataset accordingly
        train_subset = full_dataset.select(train_indices)
        val_subset = full_dataset.select(val_indices)

        # Data collator for dynamic padding
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        # Preprocess datasets
        print("⚙️ Preprocessing train_subset...")
        train_subset = train_subset.map(
            lambda batch: prepare_dataset(batch, processor),
            remove_columns=["audio_bytes"],
            num_proc=1,
            desc="Preprocessing Train"
        )
        print("⚙️ Preprocessing val_subset...")
        val_subset = val_subset.map(
            lambda batch: prepare_dataset(batch, processor),
            remove_columns=["audio_bytes"],
            num_proc=1,
            desc="Preprocessing Val"
        )

        torch.cuda.empty_cache()

        print(f"📊 Train samples: {len(train_subset)}")
        print(f"📊 Validation samples: {len(val_subset)}")

        # ---------- callbacks dinámicos ----------
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
        if is_rank0():                       # solo rank 0 usa MLflowCallback
            callbacks.append(MLflowCallback())

        # Initialize Hugging Face Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            tokenizer=processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor),
            callbacks=callbacks
        )

        print(f"\n🏋️ Training on Fold {fold_idx+1}...")
        # ---------- MLflow handled only by rank-0 -----------------------------------
        if is_rank0():
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
            mlflow.set_experiment(cfg["mlflow_experiment"])
            run_ctx = mlflow.start_run(run_name=f"fold_{fold_idx + 1}")   # returns a context manager
        else:
            from contextlib import nullcontext
            run_ctx = nullcontext()                                       # no-op for non-zero ranks
        # ---------------------------------------------------------------------------

        # Train (all ranks execute; only rank-0 logs to MLflow)
        with run_ctx:
            with tqdm(total=training_args.num_train_epochs,
                    desc=f"🏋️ Fold {fold_idx + 1}") as pbar:
                trainer.train()
                pbar.update(1)  # update once per epoch or adapt as you wish
            # optional custom logging — executed only by rank-0

        # 3) Calcular CER para este fold y hacer checkpoint
            cer_list = compute_cer_per_fold(model, processor, val_subset, fold_idx)
        cer_records.extend(cer_list)
        # Guardar checkpoint parcial inmediatamente
        with open(cer_file, "w") as f:
            json.dump(cer_records, f, indent=2)
        print(f"💾 Fold {fold_idx+1} completo — {len(cer_list)} CERs guardados.")

        del model
        del processor
        torch.cuda.empty_cache()

        # Delete checkpoints
        checkpoints_dir = cfg.get("output_dir", "./results")
        if os.path.exists(checkpoints_dir):
            print(f"🧹 Removing checkpoint directory after fold {fold_idx+1}...")
            shutil.rmtree(checkpoints_dir, ignore_errors=True)

        print("🧹 Removed IDs so far:", list(cer_records))
    os.rename(cer_file, cer_file_done)
    print(f"\n✅ Cross-validation terminado. Archivo final: {cer_file_done}")
if __name__ == "__main__":
    train()
"""
utils.py

Utility functions for evaluating and filtering audio transcription data using Whisper.
Includes metric computation, per-sample CER scoring, dynamic filtering by strategy,
and logging of removed examples.

Configuration is loaded from `experiment_config.json`.

Author: [Your Name or Lab]
"""

import evaluate
import jiwer
import json
import numpy as np
from tqdm.auto import tqdm
from transformers import TrainerCallback
from dataclasses import dataclass
from typing import Any, Dict

# 📏 Load standard metrics for speech evaluation
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred, processor):
    """
    Compute WER and CER for a given prediction batch.

    Args:
        pred: Trainer prediction output (includes .predictions and .label_ids)
        processor: Whisper processor (includes tokenizer)

    Returns:
        Dictionary with WER and CER percentages.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace ignored tokens (-100) with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode both prediction and reference
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


def minimal_transform(text):
    """
    Performs minimal tokenization to check if text is empty.

    Args:
        text (str): Input string.

    Returns:
        list: List of words or [""] if text is empty.
    """
    words = text.split()
    return [""] if not words else words


def compute_cer_sample(example):
    """
    Computes CER for a single example and detects empty hypotheses.

    Args:
        example (dict): Dataset row with 'reference' and 'hypothesis'.

    Returns:
        dict: The example with 'cer' field added.
    """
    # Ensure hypothesis key exists
    if "hypothesis" not in example:
        print("Missing 'hypothesis' in example ID:", example.get("id"))
        example["hypothesis"] = ""

    reference = str(example["reference"]).strip()
    hypothesis = str(example["hypothesis"]).strip()

    # Detect and log empty hypotheses
    hyp_words = minimal_transform(hypothesis)
    if hyp_words == [""]:
        log_file = "empty_hypothesis_ids.json"
        try:
            with open(log_file, "r") as f:
                empty_ids = json.load(f)
        except FileNotFoundError:
            empty_ids = []
        empty_ids.append(example["id"])
        with open(log_file, "w") as f:
            json.dump(empty_ids, f, indent=2)

    # Calculate CER
    cer_value = jiwer.cer(reference, hypothesis)
    example["cer"] = cer_value

    return example


def validate_and_filter(model, processor, val_subset, fold_idx, cfg):
    """
    Performs validation inference and filters out samples with high CER
    according to the strategy defined in the experiment configuration.

    Args:
        model: Trained Whisper model.
        processor: Whisper processor for tokenization/feature extraction.
        val_subset: Dataset split to validate and filter.
        fold_idx: Current fold index (used in JSON output).

    Returns:
        set: Set of IDs that were removed due to high CER.
    """
    print(f"\n🔍 Starting inference and filtering for Fold {fold_idx+1}...")

    # Inference on validation subset
    val_subset = val_subset.map(
        lambda batch: infer_batch(batch, processor, model),
        batched=True, batch_size=16, num_proc=1,
        desc="Running batch inference"
    )

    # Compute CER per sample
    print("📊 Computing CER per sample...")
    val_subset = val_subset.map(
        compute_cer_sample, num_proc=1,
        desc="Computing CER"
    )

    # Filtering logic based on config
    strategy = cfg.get("filtering_strategy", "top_percent")

    # Ensure required fields exist based on strategy
    if strategy == "threshold" and "filtering_threshold" not in cfg:
        raise ValueError("Missing 'filtering_threshold' in experiment_config.json")

    if strategy == "top_percent" and "top_percent" not in cfg:
        print("⚠️ 'top_percent' not specified. Using default: 3%")

    if strategy == "threshold":
        threshold_value = cfg["filtering_threshold"]
        removed_ids = [ex["id"] for ex in val_subset if ex["cer"] > threshold_value]
        print(f"📉 Filtering samples with CER > {threshold_value} | Removed: {len(removed_ids)}")

    elif strategy == "top_percent":
        top_percent = cfg.get("top_percent", 0.03)
        all_cers = val_subset["cer"]
        threshold_value = np.quantile(all_cers, 1 - top_percent)
        removed_ids = [ex["id"] for ex in val_subset if ex["cer"] > threshold_value]
        print(f"📉 Filtering top {top_percent*100:.1f}% by CER | Threshold: {threshold_value:.4f} | Removed: {len(removed_ids)}")

    else:
        raise ValueError(f"Unsupported filtering strategy: {strategy}")

    # Save filtered IDs to JSON log
    log_file = cfg["json_output"]
    try:
        with open(log_file, "r") as f:
            removed_ids_overall = json.load(f)
    except FileNotFoundError:
        removed_ids_overall = {}

    removed_ids_overall[f"fold_{fold_idx+1}"] = removed_ids
    with open(log_file, "w") as f:
        json.dump(removed_ids_overall, f, indent=2)

    print("\n📜 🔍 Removed IDs so far:")
    print(json.dumps(removed_ids_overall, indent=2))

    return set(removed_ids)


def infer_batch(batch, processor, model):
    """
    Performs model inference on a batch of features.

    Args:
        batch (dict): Batch of input features (preprocessed audio).
        processor: Whisper processor.
        model: Whisper model (trained or pretrained).

    Returns:
        dict: Dictionary with 'hypothesis' field containing predictions.
    """
    inputs = processor.feature_extractor.pad(
        [{"input_features": feat} for feat in batch["input_features"]],
        return_tensors="pt"
    )
    inputs = inputs.input_features.to(model.device)

    # Generate predictions
    generated_ids = model.generate(inputs)

    # Decode to text
    hypotheses = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return {"hypothesis": hypotheses}


def compute_cer_per_fold(model, processor, val_subset, fold_idx):
    """
    Ejecuta inferencia y calcula CER para cada muestra en val_subset.
    Devuelve lista de dicts: {"sample_id","fold","cer"}.
    """
    # 1) Inferencia batch a batch
    val_subset = val_subset.map(
        lambda batch: infer_batch(batch, processor, model),
        batched=True, batch_size=16, num_proc=1,
        desc="🔍 Inferencia de hipótesis"
    )

    # 2) Calcular CER usando tu función existente
    val_subset = val_subset.map(
        compute_cer_sample,
        num_proc=1,
        desc="✂️ Cálculo de CER"
    )

    # 3) Empaquetar resultados en una lista de diccionarios
    cer_list = [
        {"sample_id": ex["id"], "fold": fold_idx, "cer": ex["cer"]}
        for ex in val_subset
    ]

    # 4) Devolver la lista de métricas CER
    return cer_list



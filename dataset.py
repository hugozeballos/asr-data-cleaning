"""
dataset.py

Dataset loading, preprocessing, and splitting utilities for Whisper training and evaluation.
Includes audio decoding from byte format, Whisper-style feature extraction, and 10-fold
cross-validation preparation with support for debug/full mode.

Author: [Your Name or Lab]
"""

import io
import json
import numpy as np
import torch
import soundfile as sf
import torchaudio
from datasets import load_dataset, concatenate_datasets
from dataclasses import dataclass
from typing import Any, Dict
from tqdm.auto import tqdm


def decode_audio(audio_bytes):
    """
    Decodes audio from byte array using soundfile.

    Args:
        audio_bytes (bytes): Raw audio bytes.

    Returns:
        waveform (np.array): Decoded audio waveform.
        sample_rate (int): Sampling rate of the audio.
    """
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file, dtype="float32")

    # If stereo, take only the first channel
    if waveform.ndim > 1:
        waveform = waveform[:, 0]

    return waveform, sample_rate


def prepare_dataset(batch, processor):
    """
    Applies Whisper preprocessing (feature extraction and tokenization) to a data sample.

    Args:
        batch (dict): A dataset sample with 'audio_bytes' and 'reference'.
        processor: HuggingFace Whisper processor (includes tokenizer and feature extractor).

    Returns:
        dict: Batch with 'input_features' and 'labels' added.
    """
    # Decode audio depending on its format
    if isinstance(batch["audio_bytes"], bytes):
        audio, sampling_rate = decode_audio(batch["audio_bytes"])
    else:
        audio = batch["audio_bytes"]["array"]
        sampling_rate = batch["audio_bytes"]["sampling_rate"]

    # Extract Whisper input features
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]

    # Tokenize reference transcript
    batch["labels"] = processor.tokenizer(batch["reference"]).input_ids

    return batch


def load_datasets():
    """
    Loads and trims the dataset from Hugging Face Hub.

    Returns:
        Tuple of (train_dataset, validation_dataset), each with essential fields only.
    """
    dataset_name = cfg.get("dataset_name")
    dataset = load_dataset(dataset_name)    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    needed_columns = ['id', 'audio_bytes', 'reference']
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_columns])
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in needed_columns])

    return train_dataset, val_dataset


def prepare_dataset_for_cross_validation(cfg):
    """
    Loads and merges train/validation sets, assigns original IDs,
    and splits into 10 folds for cross-validation.

    - If mode == "debug", limits to a small sample for testing.
    - If mode == "full" (or omitted), uses the complete dataset.

    Returns:
        full_dataset (Dataset): Preprocessed dataset with 'original_id' column.
        folds (List[np.array]): List of 10 arrays with fold indices.
    """
    mode = cfg.get("mode", "full")

    # Load complete train + validation set
    dataset_name = cfg.get("dataset_name")
    dataset = load_dataset(dataset_name)
    full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

    if mode == "debug":
        sample_size = min(20, len(full_dataset))
        full_dataset = full_dataset.select(range(sample_size))
        print(f"🧪 DEBUG mode: Using a sample of {sample_size} examples.")
    else:
        print(f"🚀 FULL mode: Using entire dataset of {len(full_dataset)} examples.")

    # Assign reproducible IDs to each sample
    full_dataset = full_dataset.add_column("original_id", [f"sample_{i}" for i in range(len(full_dataset))])

    # Create 10 folds using shuffled indices
    total_size = len(full_dataset)
    indices = np.arange(total_size)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = total_size // 10
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(10)]

    return full_dataset, folds


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom collator for Whisper speech-to-text training.
    Pads both input features and target labels properly for batch training.

    Attributes:
        processor: HuggingFace Whisper processor.
    """
    processor: Any

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        """
        Applies padding and batching to a list of features.

        Args:
            features (list): List of examples with 'input_features' and 'labels'.

        Returns:
            Dict[str, torch.Tensor]: Batched and padded tensors for training.
        """
        # Extract input features and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]

        batch = {}

        # Pad input features using feature extractor
        batch["input_features"] = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        ).input_features

        # Pad labels using tokenizer
        batch["labels"] = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        ).input_ids

        return batch

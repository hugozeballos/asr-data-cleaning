import io
import numpy as np
import json
import torchaudio
import soundfile as sf
from datasets import load_dataset
from tqdm.auto import tqdm

def decode_audio(audio_bytes):
    """Decodifica el audio desde bytes."""
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return waveform, sample_rate

def prepare_dataset(batch, processor):
    """Preprocesa el dataset con Whisper."""
    if isinstance(batch["audio_bytes"], bytes):
        audio, sampling_rate = decode_audio(batch["audio_bytes"])
    else:
        audio = batch["audio_bytes"]["array"]
        sampling_rate = batch["audio_bytes"]["sampling_rate"]

    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    batch["labels"] = processor.tokenizer(batch["reference"]).input_ids
    return batch

def load_datasets():
    """Carga el dataset desde Hugging Face y lo preprocesa."""
    dataset = load_dataset("HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics")
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    needed_columns = ['id', 'audio_bytes', 'reference']
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_columns])
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in needed_columns])

    return train_dataset, val_dataset


import numpy as np
from datasets import concatenate_datasets


def prepare_dataset_for_cross_validation(sample_size=20):
    """Une train y val, asigna IDs y los divide en 10 partes."""
    dataset = load_dataset("HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics")

    # Unir train y val manteniendo el orden
    #full_dataset = dataset["train"] + dataset["validation"]
    full_dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

        # Tomar solo una fracción pequeña del dataset para pruebas
    #sample_size = min(sample_size, len(full_dataset))  # Asegurar que sample_size no exceda el dataset
    #full_dataset = full_dataset.select(range(sample_size))

    # Asignar ID único basado en el orden original
    full_dataset = full_dataset.add_column("original_id", [f"sample_{i}" for i in range(len(full_dataset))])
 
    # Dividir en 10 partes asegurando que los IDs se mantengan constantes
    total_size = len(full_dataset)
    indices = np.arange(total_size)
    np.random.seed(42)  # Para reproducibilidad
    np.random.shuffle(indices)
    fold_size = total_size // 10
    folds = [indices[i * fold_size: (i + 1) * fold_size] for i in range(10)]

    return full_dataset, folds


from dataclasses import dataclass
from typing import Any, Dict
import torch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        """Collator para padding en lotes de entrenamiento"""
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]
        batch = {}

        batch["input_features"] = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        ).input_features

        batch["labels"] = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        ).input_ids

        return batch

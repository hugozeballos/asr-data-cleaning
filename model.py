"""
model.py

Loads the Whisper model for training or inference.
The model name is read from the experiment configuration file.

Author: [Your Name or Institution]
"""

import json
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def load_experiment_config(path="experiment_config.json"):
    """Loads experiment configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


def load_model():
    """
    Loads a Whisper model and its corresponding processor from Hugging Face,
    using the model name specified in `experiment_config.json`.

    Returns:
        model (PreTrainedModel): Whisper model ready for training/evaluation
        processor (WhisperProcessor): Feature extractor + tokenizer
    """
    cfg = load_experiment_config()
    model_name = cfg["model_name"]

    print(f"📦 Loading model from: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    return model, processor

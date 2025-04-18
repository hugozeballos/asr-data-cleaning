"""
model.py

Loads the Whisper model for training or inference.
The model name is read from the experiment configuration file.

Author: [CENIA]
"""

from transformers import WhisperForConditionalGeneration, WhisperProcessor

def load_model(cfg):
    """
    Loads a Whisper model and its corresponding processor from Hugging Face,
    using the model name specified in `experiment_config.json`.

    Returns:
        model (PreTrainedModel): Whisper model ready for training/evaluation
        processor (WhisperProcessor): Feature extractor + tokenizer
    """
    model_name = cfg["model_name"]

    print(f"📦 Loading model from: {model_name}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    return model, processor

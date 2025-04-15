import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

def load_model(model_name="openai/whisper-large-v3-turbo"):
    """Carga el modelo Whisper y el procesador."""
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    return model, processor

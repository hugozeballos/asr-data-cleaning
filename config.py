"""
config.py

Defines training hyperparameters for the Seq2SeqTrainer, loading them
dynamically from the experiment_config.json file.

Author: [CENIA]
"""

import json
from transformers import Seq2SeqTrainingArguments


def load_experiment_config(path="experiment_config.json"):
    """Load experiment configuration from a JSON file."""
    with open(path) as f:
        return json.load(f)


cfg = load_experiment_config()

# Training arguments for HuggingFace Seq2SeqTrainer
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",  # could be changed too if needed
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=20,
    logging_steps=5,
    learning_rate=cfg["learning_rate"],
    num_train_epochs=cfg["epochs"],
    predict_with_generate=True,
    fp16=True,
    generation_max_length=225,
    report_to="mlflow",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)

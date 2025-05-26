"""
config.py

Defines training hyperparameters for the Seq2SeqTrainer,
based dynamically on the experiment configuration.

Author: [CENIA]
"""

from transformers import Seq2SeqTrainingArguments

def get_training_args(cfg):
    """
    Builds the training arguments for HuggingFace Seq2SeqTrainer
    based on the experiment configuration.

    Args:
        cfg (dict): Experiment configuration dictionary.

    Returns:
        Seq2SeqTrainingArguments: Training arguments object.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.get("output_dir", "./results"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        evaluation_strategy=cfg.get("evaluation_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 5),
        logging_steps=cfg.get("logging_steps", 5),
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        predict_with_generate=True,
        fp16=cfg.get("fp16", True),
        generation_max_length=cfg.get("generation_max_length", 225),
        report_to=cfg.get("report_to", "mlflow"),
        load_best_model_at_end=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        dataloader_num_workers=cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=True
    )
    return training_args
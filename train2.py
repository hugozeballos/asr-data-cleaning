# train.py
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import mlflow
import mlflow.pytorch
import config
from model import load_model_and_collator

def run_training(train_dataset, validation_dataset, processor, output_dir, train_kwargs):
    """
    Función centralizada para entrenar el modelo con el dataset y processor proporcionados.
    Retorna el modelo entrenado.
    """
    device_map = config.DEVICE
    model, data_collator = load_model_and_collator(processor, device_map)

    # Si se usa LoRA, se envuelve el modelo (ajusta según tu configuración)
    if getattr(config, "USE_LORA", False):
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # Ajusta según el tipo de tarea
            inference_mode=False,
            r=config.LORA_PARAMS.get("r", 8),
            lora_alpha=config.LORA_PARAMS.get("lora_alpha", 32),
            lora_dropout=config.LORA_PARAMS.get("lora_dropout", 0.1),
            target_modules=config.LORA_PARAMS.get("target_modules", ["q_proj", "v_proj"])
        )
        model = get_peft_model(model, lora_config)
        print("LoRA activado. Parámetros entrenables:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    training_args = Seq2SeqTrainingArguments(
        **config.TRAINING_ARGS,
        output_dir=output_dir,
        remove_unused_columns=False,  # Usar todas las columnas del dataset
    )

    mlflow.set_tracking_uri("https://mlflow-server-muiutdydxq-uc.a.run.app/")
    mlflow.set_experiment("whisper-validate-data")
    with mlflow.start_run() as run:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=lambda pred: config.compute_metrics(pred, processor.tokenizer)
            if hasattr(config, "compute_metrics") else {},
            tokenizer=processor.feature_extractor,
        )
        trainer.train()
        # Guardar el processor para reproducibilidad
        processor.save_pretrained(training_args.output_dir)
        # Loguear parámetros de entrenamiento
        for key, value in training_args.to_dict().items():
            mlflow.log_param(key, value)
    
    return model

if __name__ == "__main__":
    # Aquí se puede testear el entrenamiento de forma aislada.
    from dataset import load_and_prepare_dataset
    dataset, processor = load_and_prepare_dataset("data/dataset")
    model = run_training(dataset["train"], dataset["validation"], processor, config.TRAINING_ARGS["output_dir"], train_kwargs={})

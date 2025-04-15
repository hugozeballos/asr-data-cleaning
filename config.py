from transformers import Seq2SeqTrainingArguments

# Configurar parámetros del entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    num_train_epochs=1,  # Usar solo 1 época para prueba rápida
    fp16=True,
    eval_steps=20,
    logging_steps=5,
    learning_rate=5e-5,
    predict_with_generate=True,
    generation_max_length=225,
    report_to="mlflow",  # Evita que Colab intente reportar a MLflow
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    dataloader_num_workers=4,
    dataloader_pin_memory=True
)


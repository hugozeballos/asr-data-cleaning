import mlflow
import os
import io
import json
import torch
import torchaudio
from datasets import load_dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
from dataclasses import dataclass
from typing import Any, Dict
import jiwer
import multiprocessing as mp
import evaluate
from transformers import WhisperTokenizer
from transformers import TrainerCallback
# Mover la definición del DataCollator fuera de main()
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: list) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        labels = [feature["labels"] for feature in features]
        batch = {}
        batch["input_features"] = self.processor.feature_extractor.pad(input_features, return_tensors="pt").input_features
        batch["labels"] = self.processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt").input_ids
        return batch

# Funciones globales (si es necesario, también pueden definirse fuera de main)
def decode_audio(audio_bytes):
    import soundfile as sf
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return waveform, sample_rate

def prepare_dataset(batch, processor):
    if isinstance(batch["audio_bytes"], bytes):
        audio, sampling_rate = decode_audio(batch["audio_bytes"])
    else:
        audio = batch["audio_bytes"]["array"]
        sampling_rate = batch["audio_bytes"]["sampling_rate"]
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
    batch["labels"] = processor.tokenizer(batch["reference"]).input_ids
    return batch

def infer_batch(batch, processor, model):
    inputs = processor.feature_extractor.pad(
        [{"input_features": feat} for feat in batch["input_features"]],
        return_tensors="pt"
    )
    inputs = inputs.input_features.to(model.device)
    generated_ids = model.generate(inputs)
    hypotheses = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {"hypothesis": hypotheses}

def minimal_transform(text):
    words = text.split()
    if not words:
        return [""]
    return words

global_processor = None
    #7 Calcular CER
def compute_cer_sample(example):
    if "hypothesis" not in example:
        print("Clave 'hypothesis' no encontrada en el ejemplo con id:", example.get("id"))
        example["hypothesis"] = ""
    reference = str(example["reference"]).strip()
    hypothesis = str(example["hypothesis"]).strip()

    # Aplicar la transformación mínima a la referencia
    ref_words = minimal_transform(hypothesis)
    # Opcional: Si ref_words es [""] significa que la referencia está vacía y se guarda el id
    if ref_words == [""]:
        log_file = "empty_reference_ids.json"
        try:
            with open(log_file, "r") as f:
                empty_ids = json.load(f)
        except FileNotFoundError:
            empty_ids = []
        empty_ids.append(example["id"])
        with open(log_file, "w") as f:
            json.dump(empty_ids, f, indent=2)


    # Usar minimal_transform como transformación
    cer_value = jiwer.cer(
        reference,
        hypothesis,
        #truth_transform=minimal_transform,
        #hypothesis_transform=minimal_transform
    )
    example["cer"] = cer_value
    return example

# Definir la función compute_metrics para evaluación global
metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = global_processor.tokenizer.pad_token_id
    pred_str = global_processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = global_processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}

def main():
    # 1. Cargar modelo y procesador
    model_name = "openai/whisper-small"
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    #processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")

    global global_processor
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")
    global_processor = processor  # Asigna el processor globalmente

    # 2. Cargar el dataset ya dividido en "train" y "validation"
    dataset = load_dataset("HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics")
    train_dataset = dataset["train"] #.select(range(4))
    val_dataset = dataset["validation"] #.select(range(1))

    if "id" not in train_dataset.column_names:
        train_dataset = train_dataset.add_column("id", [f"train_{i}" for i in range(len(train_dataset))])
    if "id" not in val_dataset.column_names:
        val_dataset = val_dataset.add_column("id", [f"validation_{i}" for i in range(len(val_dataset))])

    # Conservar las columnas necesarias para la inferencia
    needed_columns = ['id', 'audio_bytes', 'reference']
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_columns])
    val_dataset = val_dataset.remove_columns([col for col in val_dataset.column_names if col not in needed_columns])

    # Preprocesamiento
    #train_dataset = train_dataset.select(range(4))
    #val_dataset = val_dataset.select(range(1))
    train_dataset = train_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=["audio_bytes"], num_proc=4)
    val_dataset = val_dataset.map(lambda batch: prepare_dataset(batch, processor), remove_columns=["audio_bytes"], num_proc=4)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Configurar TrainingArguments, incluyendo dataloader_pin_memory y dataloader_num_workers
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy="steps",
        num_train_epochs=3,
        fp16=True,
#        #save_steps=4,
        eval_steps=100,
        logging_steps=100,
        learning_rate=5e-5,
        predict_with_generate=True,
        generation_max_length=225,
#        save_total_limit=2,
        report_to="mlflow",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True
    )

    # Callback para monitorear CPU y GPU
    class CPUUsageCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            print(f"CPU usage at step {state.global_step}: {cpu_usage}%")
            if logs is not None:
                logs["cpu_usage"] = cpu_usage
            return control

    class GPUUsageCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"GPU usage at step {state.global_step}: Allocated: {mem_allocated:.2f} MB, Reserved: {mem_reserved:.2f} MB")
                if logs is not None:
                    logs["gpu_memory_allocated_mb"] = mem_allocated
                    logs["gpu_memory_reserved_mb"] = mem_reserved
            return control

    # Configurar el Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), CPUUsageCallback(), GPUUsageCallback()]
    )

    num_iterations = 3
    # Archivo para guardar los IDs removidos a lo largo de las iteraciones
    for iteration in range(num_iterations):
        print(f"\n=== Iteración {iteration+1}: Entrenamiento con {len(train_dataset)} muestras ===")

        mlflow.set_tracking_uri("https://mlflow-server-muiutdydxq-uc.a.run.app/")
        mlflow.set_experiment("whisper-validate-data-ventress2")
        trainer.train()
        mlflow.end_run()

        print("Infiriendo...")
        train_dataset = train_dataset.map(lambda batch: infer_batch(batch, processor, model), batched=True, batch_size=16, num_proc=1)
        #print("Columnas después de inferencia:", train_dataset.column_names)
        #for idx, example in enumerate(train_dataset):
        #    print(f"{idx}: Label: {example['reference']}, Hypothesis: {example['hypothesis']}")

        print("Calculando CER por muestra para el filtrado...")
        train_dataset = train_dataset.map(compute_cer_sample, num_proc=4)

        import numpy as np
        import json
        all_cers = train_dataset["cer"]
        threshold_value = np.quantile(all_cers, 0.99)
        print(f"El umbral de CER (percentil 90) es: {threshold_value:.4f}")
        removed_ids = [example["id"] for example in train_dataset if example["cer"] > threshold_value]
        print(f"Se removerán {len(removed_ids)} muestras en la iteración.")
        log_file = "removed_ids.json"
        try:
            with open(log_file, "r") as f:
                removed_ids_overall = json.load(f)
                if isinstance(removed_ids_overall, list):
                    print("el dataset es una lista")
                    # Podrías asumir que esta era la iteración 1 o simplemente reinicializarlo.
                    removed_ids_overall = {"iteracion_1": removed_ids_overall}
        except FileNotFoundError:
            removed_ids_overall = {}
        iteration_key = f"iteracion_{iteration+1}"  # Aquí podrías usar la iteración actual
        removed_ids_overall[iteration_key] = removed_ids
        with open(log_file, "w") as f:
            json.dump(removed_ids_overall, f, indent=2)
        train_dataset = train_dataset.filter(lambda example: example["cer"] <= threshold_value, num_proc=4)
        print(f"El dataset filtrado tiene {len(train_dataset)} ejemplos.")
        needed_for_training = ["input_features", "labels", "id", "reference"]
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_for_training])

    model.save_pretrained("./whisper-finetuned-es")
    processor.save_pretrained("./whisper-finetuned-es")

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()

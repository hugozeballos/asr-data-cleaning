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
)
from dataclasses import dataclass
from typing import Any, Dict
import jiwer

# 1. Cargar modelo y procesador
model_name = "openai/whisper-small"  # Puedes cambiar el tamaño del modelo
model = WhisperForConditionalGeneration.from_pretrained(model_name)
processor = WhisperProcessor.from_pretrained(model_name, language="es", task="transcribe")

# 2. Cargar el dataset ya dividido en "train" y "validation"
dataset = load_dataset("HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics")
train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# Si el dataset no tiene una columna "id", la creamos con un prefijo según el split
if "id" not in train_dataset.column_names:
    train_dataset = train_dataset.add_column("id", [f"train_{i}" for i in range(len(train_dataset))])
if "id" not in val_dataset.column_names:
    val_dataset = val_dataset.add_column("id", [f"validation_{i}" for i in range(len(val_dataset))])


# Aquí, si solo necesitas 'audio_bytes' y 'reference' para la inferencia, remueves las demás columnas:
needed_columns = ['audio_bytes', 'reference']
train_dataset = train_dataset.remove_columns(
    [col for col in train_dataset.column_names if col not in needed_columns]
)
val_dataset = val_dataset.remove_columns(
    [col for col in val_dataset.column_names if col not in needed_columns]
)

# 3. Preprocesamiento: decodificar audio y extraer features
import soundfile as sf

def decode_audio(audio_bytes):
    # Lee el audio usando soundfile desde un objeto BytesIO
    with io.BytesIO(audio_bytes) as audio_file:
        waveform, sample_rate = sf.read(audio_file, dtype="float32")
    # Si el audio es multicanal, puedes convertirlo a mono (por ejemplo, tomando el primer canal)
    if waveform.ndim > 1:
        waveform = waveform[:, 0]
    return waveform, sample_rate

def prepare_dataset(batch):
    if isinstance(batch["audio_bytes"], bytes):
        audio, sampling_rate = decode_audio(batch["audio_bytes"])
    else:
        audio = batch["audio_bytes"]["array"]
        sampling_rate = batch["audio_bytes"]["sampling_rate"]
    batch["input_features"] = processor.feature_extractor(audio, sampling_rate=sampling_rate).input_features[0]
# Se asume que la transcripción está en "sentence"
    batch["labels"] = processor.tokenizer(batch["reference"]).input_ids
    return batch

train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names, num_proc=4)
val_dataset = val_dataset.map(prepare_dataset, remove_columns=val_dataset.column_names, num_proc=4)

# 4. Definir data collator para padding dinámico
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

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# 5. Configurar los argumentos de entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,  # Simula un batch size global de 8
    evaluation_strategy="steps",
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    predict_with_generate=True,
    generation_max_length=225,
    save_total_limit=2,
)


#6 Inferir 
def infer_batch(batch):
    # Preparamos las features en un formato adecuado para el modelo.
    # Utilizamos el feature_extractor para aplicar padding dinámico.
    inputs = processor.feature_extractor.pad(
        [{"input_features": feat} for feat in batch["input_features"]],
        return_tensors="pt"
    )
    # Movemos las features al dispositivo (GPU)
    inputs = inputs.input_features.to(model.device)
    
    # Generamos las hipótesis para todo el batch
    generated_ids = model.generate(inputs)
    
    # Decodificamos las transcripciones en batch
    hypotheses = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return {"hypothesis": hypotheses}

#7 Calcular CER
def compute_cer_sample(example):
    # Definir la transformación para normalizar los textos
    if "hypothesis" not in example:
        print("Clave 'hypothesis' no encontrada en el ejemplo con id:", example.get("id"))
        example["hypothesis"] = ""  # O maneja el caso de otra forma
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces()
    ])
    cer_value = jiwer.cer(
        example["labels"],
        example["hypothesis"],
        truth_transform=transformation,
        hypothesis_transform=transformation
    )
    example["cer"] = cer_value
    return example

# Aplicar la función a todo el dataset (sin necesidad de batching en este caso, pues cada ejemplo se procesa individualmente)
#train_dataset_metric = train_dataset.map(compute_cer_sample, num_proc=4)

# 6. Preparar la métrica CER y la función para calcularla por muestra
#cer_metric = evaluate.load("cer")

#def compute_cer_sample(example):
    # Preparar la entrada y mover al dispositivo del modelo
#    input_tensor = torch.tensor(example["input_features"]).unsqueeze(0).to(model.device)
#    generated_ids = model.generate(input_tensor)
#    pred = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Calcular el CER de esta muestra individual
#    cer_value = cer_metric.compute(predictions=[pred], references=[example["sentence"]])["cer"]
#    return {"cer": cer_value}

# 7. Función de compute_metrics para evaluación global (opcional)
#def compute_metrics(eval_pred):
#    predictions, labels = eval_pred
#    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Remover los tokens -100 de las etiquetas
#    labels = [[l for l in label if l != -100] for label in labels]
#    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
#    cer = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
#    return {"cer": cer}

# 8. Flujo de filtrado iterativo
num_iterations = 3         # Número de iteraciones deseadas
train_dataset = train_dataset.select(range(8))
# Diccionario para guardar los IDs removidos
#removed_ids_overall = {"train": {}}

#calcular metricas para el entrenamiento
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decodifica las predicciones
    decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Remover los tokens -100 de las etiquetas y decodificarlas
    labels = [[l for l in label if l != -100] for label in labels]
    decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Definir la transformación para normalizar los textos
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces()
    ])
    
    # Calcular el CER para cada ejemplo y luego promediar
    cers = []
    for ref, hyp in zip(decoded_labels, decoded_preds):
        cer_value = jiwer.cer(
            ref,
            hyp,
            truth_transform=transformation,
            hypothesis_transform=transformation
        )
        cers.append(cer_value)
    
    avg_cer = sum(cers) / len(cers) if cers else 0.0
    return {"cer": avg_cer}


for iteration in range(num_iterations):
    print(f"\n=== Iteración {iteration+1}: Entrenamiento con {len(train_dataset)} muestras ===")
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Entrenar el modelo con el dataset actual
    trainer.train()
    
    
    #infiere
    print("infiriendo...")
    train_dataset = train_dataset.map(infer_batch, batched=True, batch_size=16, num_proc=4)
    print(train_dataset.column_names, flush=True)

    # Calcular el CER individual para cada muestra del dataset de entrenamiento
    print("Calculando CER por muestra para el filtrado...")
    train_dataset = train_dataset.map(compute_cer_sample, num_proc=4)
    
    # Filtrar el dataset para mantener solo aquellos ejemplos con CER <= threshold_value
    import numpy as np
    import json

    # Suponiendo que ya calculaste la columna "cer" en train_dataset

    # 1. Calcular el umbral del percentil 90
    all_cers = train_dataset["cer"]
    threshold_value = np.quantile(all_cers, 0.9)
    print(f"El umbral de CER (percentil 90) es: {threshold_value:.4f}")

    # 2. Identificar y guardar los IDs de las muestras que se removerán (CER mayor al umbral)
    removed_ids = [example["id"] for example in train_dataset if example["cer"] > threshold_value]
    print(f"Se removerán {len(removed_ids)} muestras.")

    # Guardar los IDs removidos en un archivo JSON (o en el formato que prefieras)
    with open("removed_ids.json", "w") as f:
        json.dump(removed_ids, f, indent=2)

    # 3. Filtrar el dataset para mantener solo el 90% de las muestras con menor CER
    train_dataset = train_dataset.filter(lambda example: example["cer"] <= threshold_value, num_proc=4)
    print(f"El dataset filtrado tiene {len(filtered_train_dataset)} ejemplos.")

    # Escribir el diccionario actualizado en el archivo JSON
    with open("id_remove.json", "w") as f:
        json.dump(removed_ids_overall, f, indent=2)
    
    #restablecer columnas proxima iteracion
    # Definir las columnas necesarias para el entrenamiento
    needed_for_training = ["input_features", "labels", "id"]

    # Remover todas las columnas que no estén en needed_for_training
    train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_for_training])

# 9. Guardar el modelo y procesador final
model.save_pretrained("./whisper-finetuned-es")
processor.save_pretrained("./whisper-finetuned-es")

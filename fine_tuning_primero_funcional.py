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
)
from dataclasses import dataclass
from typing import Any, Dict
import jiwer
import multiprocessing as mp
from transformers import EarlyStoppingCallback

def main():

    # 1. Cargar modelo y procesador
    model_name = "openai/whisper-small"  # Puedes cambiar el tamaño del modelo
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, task="transcribe")

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
    needed_columns = ['id', 'audio_bytes', 'reference']
    train_dataset = train_dataset.remove_columns(
        [col for col in train_dataset.column_names if col not in needed_columns]
    )
    val_dataset = val_dataset.remove_columns(
        [col for col in val_dataset.column_names if col not in needed_columns]
    )

    import psutil
    from transformers import TrainerCallback

    class CPUUsageCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            # Obtiene el uso actual de la CPU
            cpu_usage = psutil.cpu_percent(interval=None)
            print(f"CPU usage at step {state.global_step}: {cpu_usage}%")
            # También se puede agregar a los logs para enviarlo a MLflow o WandB, por ejemplo:
            if logs is not None:
                logs["cpu_usage"] = cpu_usage
            return control


    import torch

    class GPUUsageCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if torch.cuda.is_available():
                # Convertir bytes a megabytes
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f"GPU usage at step {state.global_step}: Allocated: {mem_allocated:.2f} MB, Reserved: {mem_reserved:.2f} MB")
                # Agregar estas métricas a los logs para que se reporten a MLflow u otra herramienta
                if logs is not None:
                    logs["gpu_memory_allocated_mb"] = mem_allocated
                    logs["gpu_memory_reserved_mb"] = mem_reserved
            return control

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
    train_dataset = train_dataset.select(range(4))
    val_dataset = val_dataset.select(range(1))
    #train_dataset = train_dataset.map(prepare_dataset, num_proc=4)
    #val_dataset = val_dataset.map(prepare_dataset, num_proc=4)

    train_dataset = train_dataset.map(prepare_dataset, remove_columns=["audio_bytes"], num_proc=4)
    val_dataset = val_dataset.map(prepare_dataset, remove_columns=["audio_bytes"], num_proc=4)

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
        gradient_accumulation_steps=4,  # Simula un batch size global de 8
        evaluation_strategy="steps",
        num_train_epochs=3,
        fp16=True,
        save_steps=4,
        eval_steps=2,
        logging_steps=4,
        learning_rate=5e-5,
        predict_with_generate=True,
        generation_max_length=225,
        save_total_limit=2,
        report_to="mlflow",  # Habilita autologging de métricas en MLflow
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True  # Activa el pinning de memoria para transferir datos a la GPU  # Número de workers para cargar batches asíncronamente  # Si se desea minimizar el CER
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

    def minimal_transform(text):
        words = text.split()
        if not words:
            return [""]
        return words

    #7 Calcular CER
    def compute_cer_sample(example):
        # Convertir a cadena y eliminar espacios extra
        reference = str(example["reference"]).strip()
        hypothesis = str(example["hypothesis"]).strip()
        # Definir la transformación para normalizar los textos
        if "hypothesis" not in example:
            print("Clave 'hypothesis' no encontrada en el ejemplo con id:", example.get("id"))
            example["hypothesis"] = ""  # O maneja el caso de otra forma
        #transformation = jiwer.Compose([
        #   jiwer.ToLowerCase(),
        #   jiwer.RemovePunctuation(),
        #   jiwer.RemoveMultipleSpaces()
        #])
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


        cer_value = jiwer.cer(
            reference,
            hypothesis
            #truth_transform=minimal_transform,
            #hypothesis_transform=minimal_transform
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
    # Diccionario para guardar los IDs removidos
    #removed_ids_overall = {"train": {}}

    #calcular metricas para el entrenamiento
    import evaluate
    from transformers import WhisperTokenizer

    metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer, "cer": cer}


#    def compute_metrics(eval_pred):
#        predictions, labels = eval_pred
#        # Decodifica las predicciones
#        decoded_preds = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
#        # Remover los tokens -100 de las etiquetas y decodificarlas
#        labels = [[l for l in label if l != -100] for label in labels]
#        decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#        # Definir la transformación para normalizar los textos
#        transformation = jiwer.Compose([
#            jiwer.ToLowerCase(),
#            jiwer.RemovePunctuation(),
#            jiwer.RemoveMultipleSpaces()
#        ])
#
#        # Calcular el CER para cada ejemplo y luego promediar
#        cers = []
#        for ref, hyp in zip(decoded_labels, decoded_preds):
#            cer_value = jiwer.cer(
#                ref,
#                hyp,
#                truth_transform=transformation,
#                hypothesis_transform=transformation
#            )
#            cers.append(cer_value)
#
#        avg_cer = sum(cers) / len(cers) if cers else 0.0
#        return {"cer": avg_cer}


    for iteration in range(num_iterations):
        print(f"\n=== Iteración {iteration+1}: Entrenamiento con {len(train_dataset)} muestras ===")
        mlflow.set_tracking_uri("https://mlflow-server-muiutdydxq-uc.a.run.app/")
        mlflow.set_experiment("whisper-validate-data")

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5),CPUUsageCallback(),GPUUsageCallback()]  # Por ejemplo, 2 evaluaciones sin mejora
        )

        # Entrenar el modelo con el dataset actual
        trainer.train()
        # Terminar el run de MLflow
        mlflow.end_run()

        #infiere
        print("infiriendo...")
        train_dataset = train_dataset.map(infer_batch, batched=True, batch_size=16, num_proc=1)
        print(train_dataset.column_names, flush=True)

        # Imprimir inferencias para depuración
        for idx, example in enumerate(train_dataset):
            print(f"{idx}: Label: {example['reference']}, Hypothesis: {example['hypothesis']}")

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

        # Abrir (o crear) el archivo JSON y agregar los IDs de esta iteración
        log_file = "removed_ids.json"
        try:
            with open(log_file, "r") as f:
                removed_ids_overall = json.load(f)
        except FileNotFoundError:
            removed_ids_overall = {}

        # Usar una clave que identifique la iteración, por ejemplo "iteracion_1", "iteracion_2", etc.
        iteration_key = f"iteracion_{iteration+1}"
        removed_ids_overall[iteration_key] = removed_ids

        with open(log_file, "w") as f:
            json.dump(removed_ids_overall, f, indent=2)




# Guardar los IDs removidos en un archivo JSON (o en el formato que prefieras)
        with open("removed_ids.json", "w") as f:
            json.dump(removed_ids, f, indent=2)

        # 3. Filtrar el dataset para mantener solo el 90% de las muestras con menor CER
        train_dataset = train_dataset.filter(lambda example: example["cer"] <= threshold_value, num_proc=4)
        print(f"El dataset filtrado tiene {len(train_dataset)} ejemplos.")

        # Escribir el diccionario actualizado en el archivo JSON
        #with open("id_remove.json", "w") as f:
        #    json.dump(removed_ids_overall, f, indent=2)

        #restablecer columnas proxima iteracion
        # Definir las columnas necesarias para el entrenamiento
        needed_for_training = ["input_features", "labels", "id", "reference"]

        # Remover todas las columnas que no estén en needed_for_training
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col not in needed_for_training])

    # 9. Guardar el modelo y procesador final
    model.save_pretrained("./whisper-finetuned-es")
    processor.save_pretrained("./whisper-finetuned-es")



if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()

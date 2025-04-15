# iterative_filter.py

import os

# Configurar la caché en el directorio actual (pip_cache)
cache_dir = "huggingface_cache"
os.makedirs(cache_dir, exist_ok=True)
os.environ["HF_HOME"] = cache_dir  # Para Hugging Face Hub y otros componentes
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_dir, "transformers")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")


import io
from datasets import load_dataset, Dataset
from huggingface_hub import login
import config
import mlflow
import mlflow.pytorch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor
import torchaudio
import jiwer
# Importamos funciones ya definidas en los módulos existentes.
from dataset import load_and_prepare_dataset, prepare_dataset_from_bytes  # Si fuera necesario para procesamiento adicional.
from model import load_model_and_collator  # Para cargar el modelo y el data collator.
#############################################
# Funciones Auxiliares para Filtrado e Inference
#############################################

def remove_top_percent(dataset, percent, sort_key="cer"):
    """
    Ordena el dataset por la clave sort_key (por ejemplo, "cer")
    y elimina el porcentaje indicado (por ejemplo, 1.5 o 2.0).
    """
    data_list = list(dataset)
    # Orden ascendente: los peores (mayor CER) estarán al final.
    data_list.sort(key=lambda x: x[sort_key])
    n_total = len(data_list)
    n_remove = int(n_total * (percent / 100))
    filtered_list = data_list[: n_total - n_remove]
    return Dataset.from_dict({k: [sample[k] for sample in filtered_list] for k in filtered_list[0]})

# def inference_and_compute_cer(model, dataset, processor):
#     """
#     Para cada muestra del dataset, utiliza la clave "audio_bytes" para cargar el audio,
#     realiza la inferencia con el modelo y calcula el CER comparando con "references".
#     Actualiza (o añade) las claves "predictions" y "cer".
#     """
#     new_samples = []
#     for sample in dataset:
#         audio_bytes = sample["audio_bytes"]
#         reference = sample["references"]

#         # Convertir los bytes a un objeto tipo stream y cargar con torchaudio
#         audio_stream = io.BytesIO(audio_bytes)
#         waveform, sr = torchaudio.load(audio_stream)
        
#         # En caso de que el sample rate no sea 16000, se debería re-muestrear.
#         # Aquí asumimos que ya están en 16kHz o se puede integrar una función de resampleo.
        
#         # Extraer características usando el feature_extractor del processor.
#         input_features = processor.feature_extractor(
#             waveform[0].numpy(), sampling_rate=sr, return_tensors="pt"
#         ).input_features  # Shape: (1, seq_len, feature_dim)
        
#         input_features = input_features.to(config.DEVICE)
#         # Generar la transcripción
#         generated_ids = model.generate(input_features)
#         prediction = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
#         # Calcular CER usando jiwer (o una función propia de CER)
#         cer_value = jiwer.cer(reference, prediction)
        
#         # Actualizar la muestra
#         sample["predictions"] = prediction
#         sample["cer"] = cer_value
#         new_samples.append(sample)
    
#     return Dataset.from_dict({k: [sample[k] for sample in new_samples] for k in new_samples[0]})

def inference_and_compute_cer(model, dataset, processor, batch_size=4):
    """
    Procesa el dataset en batches para realizar inferencia y calcular el CER.
    """
    new_samples = []
    data_list = list(dataset)
    
    for i in range(0, len(data_list), batch_size):
        batch_samples = data_list[i:i+batch_size]
        waveforms = []
        references = []
        
        # Para cada muestra del batch, extraemos el audio y la referencia.
        for sample in batch_samples:
            audio_bytes = sample["audio_bytes"]
            reference = sample["references"]
            
            # Convertir bytes a stream y cargar el audio con torchaudio
            audio_stream = io.BytesIO(audio_bytes)
            waveform, sr = torchaudio.load(audio_stream)
            # Suponemos que ya están en 16kHz o se puede incluir resampleo aquí si es necesario.
            waveforms.append(waveform[0].numpy())  # Usamos el primer canal
            references.append(reference)
        
        # Extraer features en batch (el feature extractor se encarga del padding)
        input_features = processor.feature_extractor(
            waveforms, sampling_rate=sr, return_tensors="pt", padding=True
        ).input_features  # Shape: (batch_size, seq_len, feature_dim)
        
        input_features = input_features.to(config.DEVICE)
        
        # Inferencia en batch
        generated_ids = model.generate(input_features)
        predictions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Calcular CER para cada muestra del batch y actualizar la muestra
        for j, sample in enumerate(batch_samples):
            prediction = predictions[j]
            cer_value = jiwer.cer(references[j], prediction)
            sample["predictions"] = prediction
            sample["cer"] = cer_value
            new_samples.append(sample)
    
    # Reconstruir el dataset filtrado
    return Dataset.from_dict({k: [sample[k] for sample in new_samples] for k in new_samples[0]})


#############################################
# Función de Entrenamiento Adaptada
#############################################

def train_whisper_iterative(train_dataset, validation_dataset, processor, output_dir, early_stopping_patience, train_kwargs):
    """
    Función adaptada a partir de train.py, que entrena el modelo utilizando
    el dataset filtrado (train_dataset y validation_dataset) y retorna el modelo entrenado.
    Se utiliza Seq2SeqTrainer y la configuración definida en config.TRAINING_ARGS.
    """
    # Configurar dispositivo
    device_map = config.DEVICE  # Se usa "cuda" o lo definido en config.
    
    # Cargar modelo y data collator a partir del processor.
    model, data_collator = load_model_and_collator(processor, device_map)
    

    if getattr(config, "USE_LORA", False):
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,  # O ajusta a TaskType.CAUSAL_LM si corresponde
            inference_mode=False,
            r=config.LORA_PARAMS.get("r", 8),
            lora_alpha=config.LORA_PARAMS.get("lora_alpha", 32),
            lora_dropout=config.LORA_PARAMS.get("lora_dropout", 0.1),
            target_modules=config.LORA_PARAMS.get("target_modules", ["q_proj", "v_proj"])
        )
        model = get_peft_model(model, lora_config)
        print("LoRA activado. Parámetros entrenables:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Actualizar los argumentos de entrenamiento si es necesario, por ejemplo, para early stopping.
    training_args = Seq2SeqTrainingArguments(
        **config.TRAINING_ARGS,
        output_dir=output_dir,
    	remove_unused_columns=False,  # <-- usar el dataset sin filtrar las columnas que no se utilizan
        # Aquí se podrían incluir argumentos específicos de early stopping si se agregan en TRAINING_ARGS.
    )
    
    # --- NUEVA SECCIÓN: Procesar el dataset para incluir "input_features" y "labels" ---
    if "input_features" not in train_dataset.column_names:
        cols_to_remove = [col for col in train_dataset.column_names if col not in ["audio_bytes", "references"]]
        train_dataset = train_dataset.map(
            lambda batch: prepare_dataset_from_bytes(batch, processor),
            remove_columns=cols_to_remove,
            num_proc=1
        )

    if validation_dataset is not None and "input_features" not in validation_dataset.column_names:
        cols_to_remove_val = [col for col in validation_dataset.column_names if col not in ["audio_bytes", "references"]]
        validation_dataset = validation_dataset.map(
            lambda batch: prepare_dataset_from_bytes(batch, processor),
            remove_columns=cols_to_remove_val,
            num_proc=1
        )
    # Configurar mlflow (ya definido en train.py)
    mlflow.set_tracking_uri("https://mlflow-server-muiutdydxq-uc.a.run.app/")
    mlflow.set_experiment("whisper-validate-data")
    
    with mlflow.start_run() as run:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            # Se utiliza compute_metrics ya definida en utils, adaptada para CER/WER.
            compute_metrics=lambda pred: config.compute_metrics(pred, processor.tokenizer) if hasattr(config, "compute_metrics") else {},
            tokenizer=processor.feature_extractor,
        )
        trainer.train()
        # Guardar el processor en el directorio de salida
        processor.save_pretrained(training_args.output_dir)
    
    return model

#############################################
# Función Principal: Iterative Filtering
#############################################

def iterative_filter_whisper(
    hf_dataset_repo="HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics",
#    hf_dataset_repo,         # Ejemplo: "HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics"
    num_iterations=3,
    first_filter_percent=1.5,
    subsequent_filter_percent=2.0,
    train_kwargs=None,
    output_model_dir="HugoZeballos/whisper_iter_final",
    output_dataset_repo="HugoZeballos/Dataset_filtrado_iterativo_final",
    test_mode=False
):
    """
    Orquesta el flujo iterativo:
      1. Carga el dataset del repositorio HF.
      2. Filtra inicialmente el train usando el CER precalculado.
      3. Para cada iteración:
           a) Entrena el modelo desde cero (usando el checkpoint base de config.MODEL_NAME_OR_PATH) con early stopping.
           b) Realiza inferencia sobre el train y recalcula el CER.
           c) Filtra eliminando el porcentaje de muestras con mayor CER.
      4. Al finalizar, sube el dataset filtrado y el modelo final a HF Hub.
      
    Retorna el dataset final y el modelo final.
    """
    if train_kwargs is None:
        train_kwargs = {}
    
    ds = load_dataset(hf_dataset_repo)
    train_dataset = ds["train"]
    validation_dataset = ds["validation"] if "validation" in ds else None

    print("Filtrado inicial: eliminando el {}% de muestras con mayor CER (precalculado)".format(first_filter_percent))
    train_dataset = remove_top_percent(train_dataset, first_filter_percent, sort_key="cer")
    
            # Si estamos en modo test, limitar el dataset a un pequeño subconjunto y forzar 1 iteración.
    if test_mode:
        print("Modo test activado: limitando el dataset a las primeras 16 muestras.")
        train_dataset = train_dataset.select(range(16))
        if validation_dataset is not None:
            validation_dataset = validation_dataset.select(range(16))
        num_iterations = 1

    # Cargar el processor (se usa para extraer features y tokenizar)
    processor = WhisperProcessor.from_pretrained(config.MODEL_NAME_OR_PATH, language=config.LANGUAGE, task=config.TASK)

    for iteration in range(1, num_iterations + 1):
        print(f"\n=== Iteración {iteration}/{num_iterations} ===")
        print("Entrenando el modelo desde cero...")
        model = train_whisper_iterative(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            processor=processor,
            output_dir=f"iter_output/iter_{iteration}",
            early_stopping_patience=3,
            train_kwargs=train_kwargs
        )
        
        print("Realizando inferencia sobre el dataset de train y recalculando CER...")
        train_dataset = inference_and_compute_cer(model, train_dataset, processor)
        
        # Guardar log de CER para cada muestra
        log_filename = f"cer_log_iter_{iteration}.csv"
        with open(log_filename, "w") as log_file:
            log_file.write("sample_ids,cer\n")
            for sample in train_dataset:
                log_file.write(f"{sample['sample_ids']},{sample['cer']}\n")
        print(f"Log de CER guardado en: {log_filename}")
        
        print("Filtrando el {}% de muestras con mayor CER (tras inferencia)...".format(subsequent_filter_percent))
        train_dataset = remove_top_percent(train_dataset, subsequent_filter_percent, sort_key="cer")
        print(f"Dataset tras filtrado: {len(train_dataset)} muestras restantes.")
    
    if not test_mode:
        print("Subiendo el dataset final a Hugging Face...")
        train_dataset.push_to_hub(output_dataset_repo)
        print(f"Dataset subido a: {output_dataset_repo}")

        print("Subiendo el modelo final a Hugging Face...")
        model.push_to_hub(output_model_dir)
        print(f"Modelo subido a: {output_model_dir}")
    else:
        print("Modo test activado: omitiendo la subida a Hugging Face Hub.")

    
    return train_dataset, model

#############################################
# Ejecución Principal
#############################################

if __name__ == "__main__":
    # Autenticarse en HF Hub (usa tu token real)
    #login("REDACTED_HF_TOKEN")
    
    # Llamada a la función iterativa con el repositorio del dataset original.
    iterative_filter_whisper(
        hf_dataset_repo="HugoZeballos/original-audio-dataset-with-inferencesofwhysper-and-metrics",
        num_iterations=3,
        first_filter_percent=1.5,
        subsequent_filter_percent=2.0,
        train_kwargs={"num_train_epochs": 20, "per_device_train_batch_size": 128},  # Ajusta según config y recursos.
        test_mode=False  # <-- Activar modo test
    )

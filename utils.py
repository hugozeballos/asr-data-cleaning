import evaluate
import torch
from transformers import TrainerCallback
from dataclasses import dataclass
from typing import Any, Dict
import jiwer
import numpy as np
import json


# Cargar métricas
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred, processor):
    """Calcula WER y CER para evaluación"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Reemplazar -100 con el token de padding para que no afecte la evaluación
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer, "cer": cer}


def minimal_transform(text):
    """Transformación mínima para limpiar el texto"""
    words = text.split()
    return [""] if not words else words

def compute_cer_sample(example):
    """Calcula CER de cada ejemplo para filtrar datos con alto error"""

    # Asegurar que la clave 'hypothesis' existe en el ejemplo
    if "hypothesis" not in example:
        print("Clave 'hypothesis' no encontrada en el ejemplo con id:", example.get("id"))
        example["hypothesis"] = ""

    reference = str(example["reference"]).strip()
    hypothesis = str(example["hypothesis"]).strip()

    # Aplicar transformación mínima
    ref_words = minimal_transform(hypothesis)

    # Si la referencia es vacía, guardar el ID para revisión
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

    # Calcular CER
    cer_value = jiwer.cer(reference, hypothesis)
    example["cer"] = cer_value

    return example


import numpy as np
import json
from tqdm.auto import tqdm

def validate_and_filter(model, processor, val_subset, fold_idx):
    """Realiza inferencia, calcula CER y filtra datos con alto error"""

    print(f"\n🔍 Inferencia y filtrado del conjunto de validación en Fold {fold_idx+1}...")

    # Inferencia sobre los datos de validación
    print("\n🔍 Inferencia y filtrado del conjunto de validación...")
    val_subset = val_subset.map(lambda batch: infer_batch(batch, processor, model), 
                                    batched=True, batch_size=16, num_proc=1, 
                                    desc="Inferencia en batch")

    # Calcular CER por cada muestra
    print("📊 Calculando CER por muestra para el filtrado...")
    val_subset = val_subset.map(compute_cer_sample, num_proc=1, 
                                    desc="Calculando CER")

    # Determinar el umbral para eliminar datos con alto CER
    all_cers = val_subset["cer"]
    threshold_value = np.quantile(all_cers, 0.97)

    # Guardar IDs de ejemplos con CER alto
    removed_ids = [example["id"] for example in val_subset if example["cer"] > threshold_value]
    print(f"📉 Umbral CER: {threshold_value:.4f} | 🚫 Eliminando {len(removed_ids)} ejemplos.")

    # 3. Filtrar ejemplos con CER > 0.3
    ##threshold_value = 0.3
    ##removed_ids = [example["id"] for example in val_subset if example["cer"] > threshold_value]
    ##print(f"📉 Umbral CER fijo: {threshold_value:.2f} | 🚫 Eliminando {len(removed_ids)} ejemplos.")

    # 4. Guardar detalles de los ejemplos eliminados (referencia, hipótesis, cer)
    ##removed_examples = [
    ##    {
    ##        "id": example["id"],
    ##        "cer": example["cer"],
    ##        "reference": example.get("reference", ""),
    ##        "hypothesis": example.get("hypothesis", "")
    ##    }
    ##    for example in val_subset
    ##    if example["cer"] > threshold_value
    ##]

    # 5. Guardar todos los ejemplos eliminados de todos los folds en un solo archivo
    ##removed_examples_file = "removed_examples_all_folds.json"
    ##try:
    ##    with open(removed_examples_file, "r") as f:
    ##        removed_examples_all_folds = json.load(f)
    ##except FileNotFoundError:
    ##    removed_examples_all_folds = {}

    ##removed_examples_all_folds[f"fold_{fold_idx+1}"] = removed_examples

    ##with open(removed_examples_file, "w") as f:
    ##    json.dump(removed_examples_all_folds, f, indent=2)




    # Guardar los IDs eliminados en JSON
    log_file = "removed_ids_hydra_large.json"
    try:
        with open(log_file, "r") as f:
            removed_ids_overall = json.load(f)
    except FileNotFoundError:
        removed_ids_overall = {}

    removed_ids_overall[f"fold_{fold_idx+1}"] = removed_ids
    with open(log_file, "w") as f:
        json.dump(removed_ids_overall, f, indent=2)

        # 📢 **Mostrar el JSON después de guardarlo**
    print("\n📜 🔍 **Resumen de IDs eliminados hasta ahora:**")
    print(json.dumps(removed_ids_overall, indent=2))

    ## Filtrar el dataset completo eliminando los ejemplos de validación con alto CER
    #full_dataset = full_dataset.filter(lambda example: example["original_id"] not in removed_ids, num_proc=1)

    #print(f"✅ Dataset actualizado con {len(full_dataset)} ejemplos tras el fold {fold_idx+1}.")

    return set(removed_ids)


def infer_batch(batch, processor, model):
    """Realiza inferencia sobre un batch de datos"""
    inputs = processor.feature_extractor.pad(
        [{"input_features": feat} for feat in batch["input_features"]],
        return_tensors="pt"
    )
    inputs = inputs.input_features.to(model.device)

    # Generar predicciones con el modelo
    generated_ids = model.generate(inputs)

    # Decodificar las predicciones
    hypotheses = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    return {"hypothesis": hypotheses}

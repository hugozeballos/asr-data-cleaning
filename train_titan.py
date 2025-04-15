import mlflow
import torch
from tqdm.auto import tqdm
from transformers import Seq2SeqTrainer, EarlyStoppingCallback
from tqdm.auto import tqdm
from model_titan import load_model
from dataset import load_datasets, prepare_dataset, prepare_dataset_for_cross_validation, DataCollatorSpeechSeq2SeqWithPadding
from config import training_args
from utils import compute_metrics, compute_cer_sample, validate_and_filter, infer_batch
import numpy as np

def train():
    """Función principal para entrenar el modelo."""
    model, processor = load_model()
    train_dataset, val_dataset = load_datasets()


    # 🔹 **Usar un dataset reducido (Ejemplo: 20 muestras) para pruebas**
    full_dataset, folds = prepare_dataset_for_cross_validation(sample_size=20)


    removed_ids = set()

    # 🔹 **Validación Cruzada - 10 folds**
    for fold_idx in tqdm(range(10), desc="🔄 Progreso de Validación Cruzada"):
        print(f"\n🔄 Fold {fold_idx+1}/10 - Validación en Fold {fold_idx+1}")

        # Seleccionar índices para validación y entrenamiento según los folds originales
        val_indices = folds[fold_idx]
        train_indices = np.concatenate([folds[i] for i in range(10) if i != fold_idx])

        # Primero, seleccionar el subconjunto del dataset original según los índices
        train_subset = full_dataset.select(train_indices)
        val_subset = full_dataset.select(val_indices)


        # Preparar datasets
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        # Luego, aplicar el preprocesamiento para convertir los datos (preparar el dataset)
        print("⚙️ Preprocesando train_subset...")
        train_subset = train_subset.map(lambda batch: prepare_dataset(batch, processor),
                                    remove_columns=["audio_bytes"],
                                    num_proc=1, desc="Preprocesando Train")
        print("⚙️ Preprocesando val_subset...")
        val_subset = val_subset.map(lambda batch: prepare_dataset(batch, processor),
                                    remove_columns=["audio_bytes"],
                                    num_proc=1, desc="Preprocesando Val")

        # Finalmente, filtrar los ejemplos que ya han sido eliminados (según los IDs)
        train_subset = train_subset.filter(lambda x: x["id"] not in removed_ids, num_proc=1)
        val_subset = val_subset.filter(lambda x: x["id"] not in removed_ids, num_proc=1)

        torch.cuda.empty_cache()

        print(f"🔹 Train: {len(train_subset)} ejemplos")
        print(f"🔹 Val: {len(val_subset)} ejemplos")

        # Definir Trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            tokenizer=processor.feature_extractor,
            data_collator=data_collator,
            compute_metrics=lambda pred: compute_metrics(pred, processor),
            #remove_unused_columns=False,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
        )

        print(f"\n🚀 Entrenando en Fold {fold_idx+1} con {len(train_subset)} ejemplos...")
        mlflow.set_tracking_uri("https://mlflow-server-muiutdydxq-uc.a.run.app/")
        mlflow.set_experiment("whisper-validate-data-ventress2")
        with tqdm(total=training_args.num_train_epochs, desc=f"🏋️ Entrenando Fold {fold_idx+1}") as pbar:
            trainer.train()
            pbar.update(1)  # Avanzar en tqdm
        mlflow.end_run()

        torch.cuda.empty_cache()
        # Inferencia y filtrado: la función validate_and_filter devuelve una lista de IDs a eliminar para este fold
        new_removed_ids = validate_and_filter(model, processor, val_subset, fold_idx)

        # Actualizar el conjunto de IDs eliminados
        removed_ids.update(new_removed_ids)

        # Mostrar el JSON con los IDs eliminados hasta el momento (si se guarda en un archivo, también se podría imprimir)
        print("IDs eliminados hasta ahora:", list(removed_ids))


if __name__ == "__main__":
    train()

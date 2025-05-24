import os
import json
import random
import numpy as np
import evaluate
import mlflow
import matplotlib.pyplot as plt

from config import get_training_args
from model import load_model
from dataset import prepare_dataset_for_cross_validation
from transformers import Seq2SeqTrainer, set_seed
from utils import compute_cer_per_fold, compute_metrics

# 1. Cargar configuración
with open("comparacion_config.json") as f:
    cfg = json.load(f)

model_name = cfg["model_name"]
thresholds = cfg["thresholds"]
cer_path = cfg["cer_records_path"]
test_percent = cfg.get("test_percent", 0.05)

# 2. Leer CERs
with open(cer_path) as f:
    cer_data = json.load(f)

cer_sorted = sorted(cer_data, key=lambda x: x["cer"])
test_n = max(1, int(test_percent * len(cer_sorted)))
test_ids = {x["sample_id"] for x in cer_sorted[:test_n]}

# Mapeo rápido ID → CER
cer_map = {x["sample_id"]: x["cer"] for x in cer_data}

# 3. Cargar dataset completo
full_ds, _ = prepare_dataset_for_cross_validation(cfg)

# 4. Inicializar MLflow
mlflow.set_tracking_uri(cfg["mlflow_uri"])
mlflow.set_experiment("whisper_cer_threshold_exploration")

# 5. Métricas acumuladas
cer_global_scores = []
cer_sample_scores = {}

# 6. Iterar por threshold
for thr in thresholds:
    run_name = f"{model_name}_threshold_{thr}"
    with mlflow.start_run(run_name=run_name):

        # 6.1 Filtrar dataset excluyendo test y > threshold
        def keep_fn(ex):
            sid = ex["id"]
            return sid not in test_ids and cer_map.get(sid, 1.0) <= thr

        filtered = full_ds["train"].filter(keep_fn)

        # 6.2 Split reproducible
        seed = 42
        shuffled = filtered.shuffle(seed=seed)
        n_val = int(0.1 * len(shuffled))
        val_ds = shuffled.select(range(n_val))
        train_ds = shuffled.select(range(n_val, len(shuffled)))

        # 6.3 Preprocesamiento
        model, processor = load_model(cfg)

        train_ds = train_ds.map(lambda x: prepare_dataset(x, processor),
                                remove_columns=["audio_bytes"], num_proc=1)
        val_ds = val_ds.map(lambda x: prepare_dataset(x, processor),
                            remove_columns=["audio_bytes"], num_proc=1)

        # 6.4 Entrenamiento distribuido
        training_args = get_training_args(cfg)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=processor,
        )
        trainer.train()

        # 6.5 Evaluación
        test_ds = full_ds["train"].filter(lambda ex: ex["id"] in test_ids)
        test_ds = test_ds.map(lambda x: prepare_dataset(x, processor),
                              remove_columns=["audio_bytes"], num_proc=1)
        
        # CER por muestra (para eval_por_muestra.json y boxplot)
        cer_list = compute_cer_per_fold(model, processor, test_ds, fold_idx=0)
        cer_sample_scores[thr] = cer_list  # guardar la lista completa por threshold

        
        # CER global (real) usando compute_metrics
        preds = trainer.predict(test_ds)
        metrics = compute_metrics(preds, processor)
        cer_global = float(metrics["cer"])

        cer_global_scores.append((thr, cer_global))

        # 6.6 Guardar métricas
        eval_dir = os.path.join("comparacion", model_name, f"threshold_{thr}")
        os.makedirs(eval_dir, exist_ok=True)

        with open(os.path.join(eval_dir, "eval_por_muestra.json"), "w") as f:
            json.dump(cer_list, f, indent=2)

        with open(os.path.join(eval_dir, "eval_global.json"), "w") as f:
            json.dump({"threshold": thr, "cer_global": cer_global}, f, indent=2)

        mlflow.log_metric("cer_global", cer_global)

# 7. Graficar resultados
import matplotlib.pyplot as plt

# Scatter
ths, global_vals = zip(*cer_global_scores)
plt.figure()
plt.plot(ths, global_vals, marker='o')
plt.xlabel("Threshold CER")
plt.ylabel("CER Global")
plt.title(f"CER global vs Threshold ({model_name})")
scatter_path = os.path.join("comparacion", model_name, "scatter.png")
plt.savefig(scatter_path)
mlflow.log_artifact(scatter_path)


# Boxplot
plt.figure()
data = [[s["cer"] for s in cer_sample_scores[t]] for t in ths]
plt.boxplot(data, labels=[str(t) for t in ths])
plt.xlabel("Threshold")
plt.ylabel("CER por muestra")
plt.title(f"Distribución de CERs por umbral ({model_name})")
boxplot_path = os.path.join("comparacion", model_name, "boxplot.png")
plt.savefig(boxplot_path)
mlflow.log_artifact(boxplot_path)

print("✅ Finalizado experimento con thresholds para:", model_name)
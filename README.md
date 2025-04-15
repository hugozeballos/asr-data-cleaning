# 🧼 ASR Dataset Cleaning with Cross-Validation and Whisper

This repository implements a robust data filtering pipeline for Automatic Speech Recognition (ASR) using **Whisper-small** and **10-fold cross-validation**.  
The main goal is to evaluate and compare different data cleaning strategies, in order to identify which method produces the most reliable training dataset.

---

## 📋 Project Overview

We are not aiming to train the best model.  
Instead, we focus on answering the question:

> 💡 *Which data filtering strategy produces the cleanest and most effective dataset for ASR training?*

---

## 🧠 Methodology

### 🔁 Cross-Validation Filtering (Stage 1)
1. The dataset is split into 10 folds.
2. For each fold:
   - A Whisper-small model is trained on the other 9 folds.
   - The held-out fold is used for validation and inference.
   - Character Error Rate (CER) is computed for each sample.
   - Samples are selected for removal based on a filtering strategy:
     - **Top 3% CER**
     - **Fixed CER threshold (e.g., CER > 0.3)**
     - Other heuristics.

Each filtering strategy produces a `removed_ids_*.json` file, with structure:
```json
{
  "fold_1": ["id001", "id002", "id003"],
  "fold_2": ["id101", "id102"],
  "fold_3": ["id201", "id202"],
  "fold_4": ["id301"],
  "fold_5": ["id401", "id402", "id403"],
  "fold_6": ["id501"],
  "fold_7": ["id601", "id602"],
  "fold_8": ["id701"],
  "fold_9": ["id801", "id802", "id803"],
  "fold_10": ["id901"]
}
```
## 🧪 Evaluation of Cleaning Strategies (Stage 2)
For each filtering strategy:

1. All IDs marked for removal across folds are merged.
2. The original dataset is filtered (train, validation).
3. A small ASR model is trained on the filtered data.
4. Evaluation is performed on a common test set (5% of the original data).
5. Metrics (WER, CER), predictions and experiment metadata are stored using MLflow.

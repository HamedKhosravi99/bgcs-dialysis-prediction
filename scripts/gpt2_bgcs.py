"""BGCS with the GPT-2 screening step, following notebooks/02_gpt2_screening.ipynb.

Stages (run in order):
  finetune  cell 17: fine-tune GPT-2 (5 epochs, batch 8, warmup 500, weight decay 0.01, max length 512)
  validate  performance of the checkpoint used for screening on the held-out split
  bgcs      cell 15: 25 runs x 5,000 copula candidates -> GPT-2 screening -> classifiers,
            logging screening counts and stage timings

Refined_dataset.py (call_dataset / call_dataset_infr), imported by the notebook, is not available, so both
functions are rebuilt from serialize_row in the notebook: each record is serialized as
'feature_1: v, ..., feature_159: v. Label: y.' and the same 80/20 split (seed 42) is used.
Synthetic candidates are serialized with 'Label: 1'.

Usage: python scripts/gpt2_bgcs.py finetune validate bgcs      (SMOKE=1 for a quick end-to-end check)
"""
import glob
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorWithPadding, GPT2Model, GPT2Tokenizer, Trainer, TrainingArguments

from common import N_RUNS, OUT, RunRecorder, load_split
from run_methods import generate_gaussian_copula

SMOKE = os.environ.get("SMOKE") == "1"
GPT_DIR = os.path.join(OUT, "gpt2_smoke" if SMOKE else "gpt2")
METHOD = "bgcs_smoke" if SMOKE else "bgcs"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
NUM_CANDIDATES = 256 if SMOKE else 5000
# The notebook used batch 128 (GPU). On a 16 GB Mac that swaps; 32 gives identical outputs because every
# serialized record is truncated to exactly 512 tokens (no padding inside a batch).
INFER_BATCH = 32 if DEVICE == "mps" else 128


# ---------------------------------------------------------------- notebook code (verbatim)
class GPT2ForBinaryClassification(nn.Module):
    def __init__(self, model_name='gpt2', num_labels=1):
        super(GPT2ForBinaryClassification, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.gpt2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        # Use the last hidden state of the last token
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, -1, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.view(-1, 1).to(dtype=torch.float32))

        return {'loss': loss, 'logits': logits}


def serialize_row(row):
    features = ', '.join([f"feature_{i+1}: {row[i]}" for i in range(len(row)-1)])
    label = row[-1]
    return f"{features}. Label: {label}."


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": float((predictions == labels).mean())}  # evaluate.load("accuracy") equivalent


class InferenceDataset(TorchDataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def predict_from_dataframe(df, model, tokenizer, device='cuda', batch_size=128):
    model.eval()
    model.to(device)
    dataset = InferenceDataset(df.tolist())
    dataloader = DataLoader(dataset, batch_size=batch_size)
    all_preds, all_probs = [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs['logits']
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            clear_mps_cache()
    all_preds = torch.cat(all_preds).numpy()
    all_probs = torch.cat(all_probs).numpy()
    print(f"Number of positive predictions (label = 1): {(all_preds == 1).sum()}", flush=True)
    return all_preds, all_probs


# ---------------------------------------------------------------- memory settings (16 GB Mac)
# Eager attention plus clearing the MPS cache after every step keeps GPU memory near 6.5 GB instead of
# 12-14 GB (which made the machine swap). Both are numerically equivalent; the model is unchanged.
def use_eager_attention(model):
    model.gpt2.config._attn_implementation = "eager"
    return model


def clear_mps_cache():
    if DEVICE == "mps":
        torch.mps.empty_cache()


from transformers import TrainerCallback


class ClearCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clear_mps_cache()

    def on_evaluate(self, args, state, control, **kwargs):
        clear_mps_cache()


# ---------------------------------------------------------------- rebuilt Refined_dataset
def _serialize(X, labels):
    rows = np.column_stack([np.asarray(X.values), np.asarray(labels)])
    return [serialize_row(list(r)) for r in rows]


def call_dataset():
    _, _, X_train, X_test, y_train, y_test = load_split()
    if SMOKE:
        X_train, y_train, X_test, y_test = X_train[:40], y_train[:40], X_test[:16], y_test[:16]
    return DatasetDict({
        "train": Dataset.from_dict({"text": _serialize(X_train, y_train), "label": y_train.astype(int).tolist()}),
        "test": Dataset.from_dict({"text": _serialize(X_test, y_test), "label": y_test.astype(int).tolist()}),
    })


def call_dataset_infr(synthetic_minority_df):
    synthetic_minority_df["text"] = _serialize(synthetic_minority_df, np.ones(len(synthetic_minority_df), dtype=int))
    return synthetic_minority_df["text"]


def last_checkpoint():
    ckpts = glob.glob(os.path.join(GPT_DIR, "results", "checkpoint-*"))
    return max(ckpts, key=lambda p: int(p.rsplit("-", 1)[1]))


def load_filter_model():
    model_path = last_checkpoint()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = use_eager_attention(GPT2ForBinaryClassification())
    model.load_state_dict(load_file(os.path.join(model_path, "model.safetensors")))
    return model, tokenizer, model_path


# ---------------------------------------------------------------- stages
def finetune():
    os.makedirs(GPT_DIR, exist_ok=True)
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True, max_length=512, return_tensors='pt')

    data = call_dataset()
    sample = data["train"]["text"][:200]
    lengths = [len(tokenizer(t)["input_ids"]) for t in sample]
    tokenized_data = data.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = use_eager_attention(GPT2ForBinaryClassification())
    training_args = TrainingArguments(
        output_dir=os.path.join(GPT_DIR, "results"),
        num_train_epochs=5,
        # Effective batch 8 as in the notebook, run as 4 x 2 gradient accumulation to halve GPU memory
        # (same optimisation steps and schedule: 433 steps/epoch, 2,165 in total).
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(GPT_DIR, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_only_model=True,
        report_to="none",
        seed=42,
        max_steps=2 if SMOKE else -1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[ClearCacheCallback()],
    )
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0
    eval_result = trainer.evaluate(eval_dataset=tokenized_data["test"])
    info = {
        "device": DEVICE, "machine": platform.platform(), "processor": platform.processor(),
        "torch": torch.__version__, "train_rows": len(data["train"]), "eval_rows": len(data["test"]),
        "train_seconds": train_seconds, "final_eval": eval_result,
        "serialized_length_tokens_sample200": {"median": float(np.median(lengths)), "min": int(min(lengths)),
                                               "max": int(max(lengths)), "share_over_512": float(np.mean(np.array(lengths) > 512))},
        "log_history": trainer.state.log_history,
    }
    with open(os.path.join(GPT_DIR, "finetune_info.json"), "w") as f:
        json.dump(info, f, indent=1, default=str)
    print(f"[finetune] done in {train_seconds / 60:.1f} min; eval loss {eval_result.get('eval_loss')}", flush=True)
    del trainer, model
    if DEVICE == "mps":
        torch.mps.empty_cache()


def validate():
    from sklearn.metrics import average_precision_score, roc_auc_score
    model, tokenizer, model_path = load_filter_model()
    data = call_dataset()
    out = {"checkpoint": os.path.basename(model_path)}
    for split in ["test", "train"]:
        texts = pd.Series(data[split]["text"])
        y = np.array(data[split]["label"])
        t0 = time.time()
        preds, probs = predict_from_dataframe(texts, model, tokenizer, device=DEVICE, batch_size=INFER_BATCH)
        preds, probs = preds[:, 0], probs[:, 0]
        tp = int(((preds == 1) & (y == 1)).sum()); fn = int(((preds == 0) & (y == 1)).sum())
        fp = int(((preds == 1) & (y == 0)).sum()); tn = int(((preds == 0) & (y == 0)).sum())
        out[split] = {
            "n": int(len(y)), "positives": int(y.sum()), "seconds": time.time() - t0,
            "auroc": roc_auc_score(y, probs) if 0 < y.sum() < len(y) else None,
            "auprc": average_precision_score(y, probs) if 0 < y.sum() < len(y) else None,
            "accuracy": (tp + tn) / len(y), "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None, "precision": tp / (tp + fp) if tp + fp else None,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "prob_quantiles": {q: float(np.quantile(probs, q)) for q in [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]},
        }
        np.save(os.path.join(GPT_DIR, f"probs_{split}.npy"), probs)
    with open(os.path.join(GPT_DIR, "validation.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
    print("[validate]", json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ['auroc', 'auprc', 'sensitivity', 'specificity']}) for k, v in out.items()}, default=str), flush=True)


def bgcs():
    _, train_df, X_train, X_test, y_train, y_test = load_split()
    model, tokenizer, model_path = load_filter_model()
    model.to(DEVICE).eval()
    rec = RunRecorder(METHOD, y_test)
    feat_cols = list(X_train.columns)
    log = []
    cand_store, prob_store = {}, {}

    np.random.seed(0)
    minority_data = train_df[train_df['dialysis'] == 1].drop(columns=['dialysis'])
    marginal_probs = minority_data.mean().values
    cor_matrix = minority_data.corr().values
    num_synthetic_samples = NUM_CANDIDATES
    n_runs = 1 if SMOKE else N_RUNS

    for i in range(n_runs):
        t0 = time.time()
        synthetic_minority_samples = generate_gaussian_copula(cor_matrix, num_synthetic_samples, marginal_probs)
        synthetic_minority_df = pd.DataFrame(synthetic_minority_samples, columns=minority_data.columns)
        t1 = time.time()
        data = call_dataset_infr(synthetic_minority_df)
        t2 = time.time()
        preds, probs = predict_from_dataframe(data, model, tokenizer, device=DEVICE, batch_size=INFER_BATCH)
        t3 = time.time()
        cand_store[f"run{i}"] = synthetic_minority_samples.astype(np.int8)
        prob_store[f"run{i}"] = probs[:, 0].astype(np.float32)
        synthetic_minority_df['dialysis'] = preds[:, 0]
        # Screening rule (notebook cell 15 and the paper): candidates with predicted label 0 (GPT-2 probability <= 0.5)
        # are added to the training data, and those above 0.5 are discarded.
        synthetic_minority_df = synthetic_minority_df[synthetic_minority_df['dialysis'] == 0].reset_index(drop=True)
        synthetic_minority_df = synthetic_minority_df.drop(['text'], axis=1)
        synthetic_minority_df['dialysis'] = 1
        balanced_df = pd.concat([train_df, synthetic_minority_df], ignore_index=True)
        t4 = time.time()
        rec.fit_and_record(i, balanced_df.drop('dialysis', axis=1), balanced_df['dialysis'], X_test,
                           n_synthetic=len(synthetic_minority_df))
        t5 = time.time()
        p = probs[:, 0]
        log.append({
            "run": i, "candidates": num_synthetic_samples,
            "predicted_label_1": int((preds == 1).sum()), "predicted_label_0": int((preds == 0).sum()),
            "kept_for_training": len(synthetic_minority_df),
            **{f"share_prob_gt_{t}": float((p > t).mean()) for t in [0.1, 0.3, 0.5, 0.7, 0.9]},
            "copula_generation_s": t1 - t0, "serialization_s": t2 - t1, "gpt2_inference_s": t3 - t2,
            "classifier_training_s": t5 - t4,
        })
        print(f"[bgcs] run {i + 1}/{n_runs}: kept {len(synthetic_minority_df)} of {num_synthetic_samples}; "
              f"inference {t3 - t2:.0f}s", flush=True)
        pd.DataFrame(log).to_csv(os.path.join(OUT, METHOD, "filter_log.csv"), index=False)

    rec.save()
    np.savez_compressed(os.path.join(OUT, METHOD, "candidates.npz"), **cand_store, columns=np.array(feat_cols))
    np.savez_compressed(os.path.join(OUT, METHOD, "gpt2_probs.npz"), **prob_store)
    with open(os.path.join(OUT, METHOD, "run_info.json"), "w") as f:
        json.dump({"checkpoint": os.path.basename(model_path), "device": DEVICE, "machine": platform.platform()}, f, indent=1)
    m = pd.DataFrame(rec.metric_rows)
    print("[bgcs] finished. Median recall:", m.groupby("classifier")["recall"].median().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    for stage in sys.argv[1:]:
        {"finetune": finetune, "validate": validate, "bgcs": bgcs}[stage]()

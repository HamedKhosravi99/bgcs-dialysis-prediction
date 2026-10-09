"""Run the comparison methods and save per-run metrics, predictions, synthetic records and timings.

Each generator follows the notebook code it comes from (0-based cell positions in
notebooks/01_augmentation_and_evaluation.ipynb):
  real           cells 31-33  (no augmentation)
  binary_copula  cell 202     (BGCS stage 1 without the LLM, np.random.seed(0), 2,407/run)
  smote          cell 169     (SMOTE(sampling_strategy=0.8))
  gc             cell 188     (SDV GaussianCopulaSynthesizer fit once, 2,407/run)
  ctgan          cell 53      (CTGAN(epochs=38) refit each run, 2,407/run)

Usage: python scripts/run_methods.py real binary_copula smote gc ctgan [--runs N]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from common import N_RUNS, OUT, RunRecorder, load_split

N_SYN = 2407


def generate_gaussian_copula(cor_matrix, num_samples, marginal_probs):
    num_vars = cor_matrix.shape[0]
    mean = np.zeros(num_vars)
    mvn_samples = np.random.multivariate_normal(mean, cor_matrix, num_samples)
    uniform_samples = norm.cdf(mvn_samples)
    return (uniform_samples < marginal_probs).astype(int)


def run_method(method, n_runs):
    df, train_df, X_train, X_test, y_train, y_test = load_split()
    rec = RunRecorder(method, y_test)
    feat_cols = list(X_train.columns)
    synth_store = []
    timing = {"method": method, "generator_fit_seconds": None, "per_run_generation_seconds": []}
    t_start = time.time()

    if method == "real":
        rec.fit_and_record(0, X_train, y_train, X_test, lr_kwargs={}, n_synthetic=0)
        n_runs = 1

    elif method == "binary_copula":
        np.random.seed(0)
        minority_data = train_df[train_df["dialysis"] == 1].drop(columns=["dialysis"])
        marginal_probs = minority_data.mean().values
        cor_matrix = minority_data.corr().values
        for run in range(n_runs):
            t0 = time.time()
            syn = generate_gaussian_copula(cor_matrix, N_SYN, marginal_probs)
            timing["per_run_generation_seconds"].append(time.time() - t0)
            sdf = pd.DataFrame(syn, columns=minority_data.columns)
            sdf["dialysis"] = 1
            bal = pd.concat([train_df, sdf], ignore_index=True)
            synth_store.append(sdf[feat_cols].values.astype(np.int8))
            rec.fit_and_record(run, bal.drop("dialysis", axis=1), bal["dialysis"], X_test, n_synthetic=len(sdf))
            print(f"[{method}] run {run + 1}/{n_runs} done", flush=True)

    elif method == "smote":
        from imblearn.over_sampling import SMOTE
        Xtr = train_df.drop("dialysis", axis=1)
        ytr = train_df["dialysis"]
        for run in range(n_runs):
            t0 = time.time()
            X_res, y_res = SMOTE(sampling_strategy=0.8, random_state=run).fit_resample(Xtr, ytr)
            timing["per_run_generation_seconds"].append(time.time() - t0)
            n_new = len(X_res) - len(Xtr)
            synth_store.append(np.asarray(X_res.iloc[-n_new:][feat_cols]).astype(np.float32))
            rec.fit_and_record(run, X_res, y_res, X_test, n_synthetic=n_new)
            print(f"[{method}] run {run + 1}/{n_runs} done", flush=True)

    elif method == "gc":
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        minority_class = train_df[train_df["dialysis"] == 1]
        t0 = time.time()
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=minority_class)
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(minority_class)
        timing["generator_fit_seconds"] = time.time() - t0
        for run in range(n_runs):
            t0 = time.time()
            generated_df = synthesizer.sample(num_rows=N_SYN)
            timing["per_run_generation_seconds"].append(time.time() - t0)
            generated_df = generated_df[minority_class.columns]
            generated_df["dialysis"] = 1
            df11 = pd.concat([train_df, generated_df], ignore_index=True)
            synth_store.append(np.asarray(generated_df[feat_cols]).astype(np.float32))
            rec.fit_and_record(run, df11.drop("dialysis", axis=1), df11["dialysis"], X_test,
                               lr_kwargs={"random_state": 42, "max_iter": 1000}, n_synthetic=len(generated_df))
            print(f"[{method}] run {run + 1}/{n_runs} done", flush=True)

    elif method == "ctgan":
        from ctgan import CTGAN
        df20 = train_df[train_df["dialysis"] == 1]
        focused_columns = list(train_df.columns)
        ctgan = CTGAN(epochs=38)
        for run in range(n_runs):
            ctgan.set_random_state(run)
            t0 = time.time()
            ctgan.fit(df20, focused_columns)
            synthetic_data = ctgan.sample(N_SYN)
            timing["per_run_generation_seconds"].append(time.time() - t0)
            df30 = pd.concat([train_df, synthetic_data], ignore_index=True)
            synth_store.append(np.asarray(synthetic_data[feat_cols]).astype(np.float32))
            rec.fit_and_record(run, df30.drop("dialysis", axis=1), df30["dialysis"].astype(int), X_test,
                               n_synthetic=len(synthetic_data))
            print(f"[{method}] run {run + 1}/{n_runs} done", flush=True)
    else:
        raise ValueError(method)

    rec.save()
    if synth_store:
        np.savez_compressed(os.path.join(OUT, method, "synthetic.npz"),
                            **{f"run{r}": a for r, a in enumerate(synth_store)}, columns=np.array(feat_cols))
    timing["total_seconds"] = time.time() - t_start
    timing["n_runs"] = n_runs
    with open(os.path.join(OUT, method, "timing.json"), "w") as f:
        json.dump(timing, f, indent=1)
    m = pd.DataFrame(rec.metric_rows)
    print(f"[{method}] finished in {timing['total_seconds'] / 60:.1f} min. Median recall:",
          m.groupby("classifier")["recall"].median().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = N_RUNS
    if "--runs" in sys.argv:
        n = int(sys.argv[sys.argv.index("--runs") + 1])
        args = [a for a in args if a != str(n)]
    for meth in args:
        run_method(meth, n)

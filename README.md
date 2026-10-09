# BGCS: Binary Gaussian Copula Synthesis for early dialysis prediction

Code for the manuscript *Binary Gaussian Copula Synthesis: an LLM-powered data augmentation framework for early dialysis prediction in chronic kidney disease* by Hamed Khosravi, Milad Khanchi, Mobina Noori, Srinjoy Das, Mohammad Abdullah Al-Mamun and Imtiaz Ahmed (under review at PLOS ONE).

Only a small share of patients with chronic kidney disease progress to dialysis, and most EHR variables are binary. BGCS augments the dialysis class of such data in two stages.

1. **Copula stage.** A Gaussian copula built from the marginal probabilities and the Pearson (phi) correlation matrix of the training dialysis records generates synthetic binary records by thresholding.
2. **GPT-2 screening.** A GPT-2 (124M) binary classifier is fine-tuned on serialized training records. In each run it scores 5,000 copula candidates, and the screened candidates are added to the training data.

Decision Tree, Random Forest, SVM and Logistic Regression classifiers (scikit-learn defaults) are trained on the augmented data for 90-day dialysis prediction. BGCS is compared with no augmentation, SMOTE, CTGAN and the SDV Gaussian copula over 25 runs on a fixed test set.

## Repository layout

```
scripts/
  common.py            data loading, 80/20 split, classifiers, metrics (shared)
  run_methods.py       no augmentation, BGCS copula stage, SMOTE, Gaussian copula (SDV), CTGAN
  gpt2_bgcs.py         GPT-2 fine-tuning, validation and the full BGCS pipeline
  analyze.py           metric tables, confidence intervals, paired tests, calibration, decision curves
  table2_fidelity.py   feature-wise binomial proportion tests (Table 2 format)
  fig6_tree.py         Fig 6, the truncated CDSS decision tree
  summarize.py         collects everything into outputs/RESULTS_SUMMARY.md
notebooks/
  01_augmentation_and_evaluation.ipynb   original analysis notebook (outputs cleared)
  02_gpt2_screening.ipynb                original GPT-2 screening notebook (outputs cleared)
results/
  notebook_per_run_recall.csv            per-run recalls saved in notebook 01, used for the Table 3 CIs
data/                  place the data file here (not included, see data/README.md)
outputs/               created by the scripts (ignored by git)
run_all.sh             runs the whole pipeline
```

## Data

The data come from the TriNetX network under a data use agreement and cannot be shared. They are not part of this repository. See [data/README.md](data/README.md) for how to request access and for the file format the scripts expect.

## Setup

Python 3.11. With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

`python -m venv .venv` followed by `pip install -r requirements.txt` works as well. `requirements.txt` pins the exact versions used for the reported results. The GPT-2 weights are downloaded from Hugging Face on first use. The GPT-2 step runs on Apple MPS, CUDA or CPU.

## Running

```bash
./run_all.sh
```

or step by step:

| Step | Command | Time (Apple M4, 16 GB) |
|---|---|---|
| Comparison methods | `python scripts/run_methods.py real binary_copula smote gc ctgan` | about 16 min |
| GPT-2 fine-tuning | `python scripts/gpt2_bgcs.py finetune` | about 80 min |
| GPT-2 validation | `python scripts/gpt2_bgcs.py validate` | about 5 min |
| BGCS, 25 runs | `python scripts/gpt2_bgcs.py bgcs` | about 2.4 h |
| Statistics and figures | `python scripts/analyze.py` | a few minutes |
| Table 2 format | `python scripts/table2_fidelity.py` | about 1 min |
| Fig 6 | `python scripts/fig6_tree.py` | under 1 min |
| Summary | `python scripts/summarize.py` | seconds |

`python scripts/run_methods.py smote --runs 3` runs fewer runs. `SMOKE=1 python scripts/gpt2_bgcs.py finetune validate bgcs` checks the GPT-2 pipeline end to end in about a minute (2 training steps, 256 candidates, 1 run). Smoke outputs go to separate folders.

Two environment variables change the paths. `BGCS_DATA` sets the data file (default `data/merged_90_c1.xlsx`) and `BGCS_OUT` sets the output folder (default `outputs/`).

## Where the results in the paper come from

| Paper item | Source |
|---|---|
| Figs 1 to 5, Table 2 | notebook 01 (`table2_fidelity.py` recomputes Table 2 from the script runs) |
| Table 3, recall medians and ranges | notebook 01 for SMOTE, CTGAN, Gaussian copula and the BGCS copula stage (saved per-run values in `results/notebook_per_run_recall.csv`), notebook 02 for BGCS |
| Table 3, 95% CIs | `analyze.py` → `table_original_recall_ci.csv` |
| Table 4, specificity, precision, AUROC, AUPRC, O/E | `run_methods.py`, `gpt2_bgcs.py`, `analyze.py` → `table_full_metrics.csv` |
| Paired Wilcoxon tests with Holm correction | `analyze.py` → `table_paired_tests.csv` |
| Benjamini–Hochberg feature counts | `analyze.py` → `table_binomial_multiplicity.csv` |
| Decision curve analysis | `analyze.py` → `fig_decision_curves`, `RESULTS_SUMMARY.md` |
| GPT-2 validation, screening counts and timings | `gpt2_bgcs.py` → `outputs/gpt2/validation.json`, `outputs/bgcs/filter_log.csv` |
| Fig 6 | `fig6_tree.py` |
| Fig 7 | framework diagram, not generated by code |

## Notes on reproducibility

- All methods use the same 80/20 split (`random_state=42`). The test set has 866 patients, 40 of whom initiated dialysis. Synthetic records are generated from the training dialysis records only.
- The scripts follow the notebook code and cite the cells they come from (0-based cell positions). Re-running them gives the same medians as the notebooks for the Gaussian copula and the BGCS copula stage. SMOTE and CTGAN medians are within 0.025, and BGCS medians are within one test patient.
- Screening rule. As in notebook 02, candidates that GPT-2 labels 0 (probability ≤ 0.5) are added to the training data, about 2,690 of the 5,000 per run. See `gpt2_bgcs.py`.
- Each serialized record is about 960 GPT-2 tokens, and inputs are truncated to 512 tokens. GPT-2 therefore sees about the first 85 of the 159 features.
- `Refined_dataset.py`, which notebook 02 imports, is not available. `gpt2_bgcs.py` rebuilds its two functions from `serialize_row` in the notebook.
- On Apple MPS, GPT-2 uses inference batches of 32 and 4 × 2 gradient accumulation (effective batch 8, as in the notebook) to fit in 16 GB. The optimisation steps are the same as in the notebook.
- The notebooks were run on Google Colab and read the data through `files.upload()`.

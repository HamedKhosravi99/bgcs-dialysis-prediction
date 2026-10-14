#!/usr/bin/env bash
# Runs the full pipeline. Expects the data at data/merged_90_c1.xlsx (or set BGCS_DATA).
# Total time on an Apple M4 (16 GB) is about 4.5 hours, most of it GPT-2 fine-tuning and screening.
set -euo pipefail
cd "$(dirname "$0")"

python scripts/run_methods.py real binary_copula smote gc ctgan
python scripts/gpt2_bgcs.py finetune validate bgcs
python scripts/analyze.py
python scripts/table2_fidelity.py
python scripts/fig5_recall_gain.py
python scripts/fig6_tree.py
python scripts/summarize.py

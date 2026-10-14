"""Fig 5: percentage gain in median recall over training on real data alone.

Same plot as notebook 01 cell 245. The values are the Table 3 medians and the recall without augmentation.
- CTGAN, SMOTE and the Gaussian copula: medians of the per-run recalls saved in notebook 01
  (results/notebook_per_run_recall.csv).
- BGCS: Table 3 medians from notebook 02 (GPT-2 screening), whose per-run values were not saved.
- No augmentation: outputs/real/metrics.csv if run_methods.py has been run, otherwise the values reported
  in the Fig 5 caption. The models without augmentation are deterministic, so both give the same values.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from common import OUT, ROOT

MODELS = {"RF": "RandomForest", "LR": "LogisticRegression", "SVM": "SVM", "DT": "DecisionTree"}
BGCS_TABLE3 = {"RF": 0.800, "LR": 0.875, "SVM": 0.850, "DT": 0.775}
NO_AUG_PAPER = {"RF": 0.475, "LR": 0.625, "SVM": 0.575, "DT": 0.475}


def no_augmentation():
    path = os.path.join(OUT, "real", "metrics.csv")
    if not os.path.exists(path):
        return NO_AUG_PAPER
    m = pd.read_csv(path)
    return {k: float(m.loc[m["classifier"] == v, "recall"].median()) for k, v in MODELS.items()}


def main():
    nb = pd.read_csv(os.path.join(ROOT, "results", "notebook_per_run_recall.csv"))
    med = nb.groupby(["method", "classifier"])["recall"].median()
    data = {k: [no_augmentation()[k], med["ctgan", v], med["smote", v], med["gc", v], BGCS_TABLE3[k]]
            for k, v in MODELS.items()}
    index = ["Real-Data", "CTGAN", "SMOTE", "Copula", "Binary Gaussian Copula Synthesis"]
    df = pd.DataFrame(data, index=index)
    pct = (df - df.loc["Real-Data"]) / df.loc["Real-Data"] * 100
    print(df.to_string(float_format=lambda x: f"{x:.3f}"))
    print(pct.round(1).to_string())

    plt.figure(figsize=(10, 6))
    for model in pct.columns:
        plt.plot(pct.index, pct[model], marker="o", label=model)
    plt.xlabel("Methods", fontsize=14)
    plt.ylabel("Percentage Improvement (%)", fontsize=14)
    plt.legend(title="Models", loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right")
    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, "Fig5.png")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.close()
    # Same size as the published figure (1560 pixels wide at 300 dpi). PLOS figure files are RGB TIFF without alpha.
    im = Image.open(png).convert("RGB")
    im = im.resize((1560, round(im.height * 1560 / im.width)), Image.LANCZOS)
    im.save(png, dpi=(300, 300))
    im.save(os.path.join(fig_dir, "Fig5.tif"), dpi=(300, 300), compression="tiff_lzw")


if __name__ == "__main__":
    main()

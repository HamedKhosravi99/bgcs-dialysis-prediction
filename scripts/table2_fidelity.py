"""Table 2 (feature-wise binomial proportion tests) recomputed from the runs, in the paper's format.

Real dialysis records (all 241, as in notebook 01) vs each method's synthetic records, 159 features.
Statistics are computed per run and summarised by the median over the 25 runs.
"""
import os
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from common import OUT, load_split

ALPHAS = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5]
df, *_ = load_split()
real = df[df["dialysis"] == 1].drop(columns=["dialysis"])
sets = {}
for m in ["smote", "ctgan", "gc", "binary_copula"]:
    z = np.load(os.path.join(OUT, m, "synthetic.npz"))
    sets[m] = [np.asarray(z[f"run{r}"], float) for r in range(25)]
cz, pz = np.load(os.path.join(OUT, "bgcs", "candidates.npz")), np.load(os.path.join(OUT, "bgcs", "gpt2_probs.npz"))
sets["bgcs"] = [cz[f"run{r}"][pz[f"run{r}"] <= 0.5].astype(float) for r in range(25)]
rows = []
for m, runs in sets.items():
    for r, syn in enumerate(runs):
        zs, ps = [], []
        for j in range(real.shape[1]):
            with np.errstate(all="ignore"):
                zz, pp = proportions_ztest([real.iloc[:, j].sum(), syn[:, j].sum()], [len(real), len(syn)])
            zs.append(zz); ps.append(pp)
        zs, ps = np.array(zs, float), np.array(ps, float)
        ok = ~np.isnan(ps)
        zs, ps = zs[ok], ps[ok]
        rows.append({"method": m, "run": r, "features": int(ok.sum()), "z_min": zs.min(), "z_max": zs.max(), "z_mean": zs.mean(),
                     "z_sd": zs.std(ddof=1), "p_min": ps.min(), "p_max": ps.max(), "p_mean": ps.mean(), "p_sd": ps.std(ddof=1),
                     "dissimilar": int(sum((ps < a).sum() for a in ALPHAS)), "similar": int(sum((ps >= a).sum() for a in ALPHAS)),
                     "sig_raw": int((ps < 0.05).sum()), "sig_bonf": int(multipletests(ps, 0.05, "bonferroni")[0].sum()),
                     "sig_bh": int(multipletests(ps, 0.05, "fdr_bh")[0].sum())})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "analysis", "table2_fidelity_per_run.csv"), index=False)
med = res.groupby("method").median(numeric_only=True).drop(columns="run")
med = med.loc[["smote", "ctgan", "gc", "binary_copula", "bgcs"]]
med.to_csv(os.path.join(OUT, "analysis", "table2_fidelity_median.csv"))
pd.set_option("display.width", 250)
print(med.T.to_string(float_format=lambda x: f"{x:.4g}"))

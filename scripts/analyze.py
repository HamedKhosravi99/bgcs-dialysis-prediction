"""Statistics, tables and figures for the comparison of augmentation methods.

Run after run_methods.py and gpt2_bgcs.py. Writes to outputs/analysis/:
  table_full_metrics.csv           median [IQR] of every metric per method x classifier
  table_thresholds.csv             sensitivity / specificity / PPV / NPV at fixed probability thresholds
  fig_calibration.(png|tif)        reliability curves
  fig_decision_curves.(png|tif)    decision curve analysis (net benefit)
  table_bootstrap_ci.csv           95% CIs (stratified bootstrap over test patients, median over runs)
  table_paired_tests.csv           BGCS vs each method: Wilcoxon (paired by run) + Holm, Friedman,
                                   DeLong (AUROC) and exact McNemar on the shared test set
  table_original_recall_ci.csv     CIs for the per-run recalls saved in notebook 01 (Table 3),
                                   read from results/notebook_per_run_recall.csv
  table_binomial_multiplicity.csv  features with different prevalence: raw, Bonferroni, Benjamini-Hochberg
  table_ablation.csv               Gaussian Copula vs BGCS stage 1 (no LLM) vs BGCS
  table_filter_efficiency.csv      GPT-2 screening counts, acceptance and timing
summarize.py then collects these into outputs/RESULTS_SUMMARY.md.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, norm, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from common import CLASSIFIERS, DCA_THRESHOLDS, OUT, ROOT, load_split

AN = os.path.join(OUT, "analysis")
# Per-run recalls saved in the outputs of notebook 01 (cells 55, 171, 191 and 204), the values behind Table 3
SAVED_RECALLS = os.path.join(ROOT, "results", "notebook_per_run_recall.csv")
METHODS = [("real", "No augmentation"), ("smote", "SMOTE"), ("ctgan", "CTGAN"), ("gc", "Gaussian Copula"),
           ("binary_copula", "BGCS stage 1 (no LLM)"), ("bgcs", "BGCS")]
LABEL = dict(METHODS)
CLF_LABEL = {"DecisionTree": "DT", "RandomForest": "RF", "SVM": "SVM", "LogisticRegression": "LR"}
KEY_METRICS = ["recall", "specificity", "precision", "f1", "auroc", "auprc"]
ALL_METRICS = ["recall", "specificity", "precision", "npv", "f1", "balanced_accuracy", "mcc", "auroc", "auprc",
               "brier", "cal_slope", "cal_intercept", "ece", "oe_ratio"]
# Validated categorical slots (dataviz reference palette); the no-augmentation reference is neutral ink.
STYLE = {"bgcs": ("#2a78d6", "-", "o"), "binary_copula": ("#eb6834", "--", "s"), "gc": ("#1baf7a", "-.", "^"),
         "smote": ("#eda100", ":", "D"), "ctgan": ("#e87ba4", (0, (5, 1, 1, 1)), "v"),
         "real": ("#52514e", (0, (1, 1)), "x")}
B_BOOT = int(os.environ.get("B_BOOT", 2000))


# ------------------------------------------------------------------ loading
def available():
    return [m for m, _ in METHODS if os.path.exists(os.path.join(OUT, m, "metrics.csv"))]


def load_metrics(methods):
    return pd.concat([pd.read_csv(os.path.join(OUT, m, "metrics.csv")) for m in methods], ignore_index=True)


def load_preds(m):
    z = np.load(os.path.join(OUT, m, "predictions.npz"))
    return {k: z[k] for k in z.files}


def runs_matrix(metrics, m, clf, col, n=25):
    v = metrics[(metrics.method == m) & (metrics.classifier == clf)].sort_values("run")[col].to_numpy()
    return np.repeat(v, n) if len(v) == 1 else v


# ------------------------------------------------------------------ metric helpers
def average_precision(y, s):
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    idx = np.r_[np.where(np.diff(s_sorted))[0], len(y) - 1]
    tps = np.cumsum(y_sorted)[idx]
    fps = idx + 1 - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def bootstrap_ci(y, ypred, score, seed=0):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    n1, n0 = len(pos), len(neg)
    out = {k: [] for k in KEY_METRICS}
    for _ in range(B_BOOT):
        idx = np.r_[rng.choice(pos, n1), rng.choice(neg, n0)]
        P, S = ypred[:, idx], score[:, idx]
        tp = P[:, :n1].sum(1); fn = n1 - tp
        fp = P[:, n1:].sum(1); tn = n0 - fp
        yb = np.r_[np.ones(n1, int), np.zeros(n0, int)]
        ranks = rankdata(S, axis=1)
        auc = (ranks[:, :n1].sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
        ap = [average_precision(yb, S[r]) for r in range(S.shape[0])]
        vals = {"recall": tp / n1, "specificity": tn / n0,
                "precision": np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0),
                "f1": 2 * tp / np.maximum(2 * tp + fp + fn, 1), "auroc": auc, "auprc": np.array(ap)}
        for k in KEY_METRICS:
            out[k].append(np.median(vals[k]))
    return {k: (np.percentile(v, 2.5), np.percentile(v, 97.5)) for k, v in out.items()}


def delong_test(y, s1, s2):
    """Two-sided DeLong test for two correlated AUROCs (Sun & Xu fast implementation)."""
    pos, neg = s1[y == 1], s1[y == 0]
    pos2, neg2 = s2[y == 1], s2[y == 0]
    m, n = len(pos), len(neg)

    def comps(p, q):
        tx = rankdata(p); ty = rankdata(q); tz = rankdata(np.r_[p, q])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1 - (tz[m:] - ty) / m
        return auc, v01, v10

    a1, v01a, v10a = comps(pos, neg)
    a2, v01b, v10b = comps(pos2, neg2)
    sx = np.cov(np.vstack([v01a, v01b]))
    sy = np.cov(np.vstack([v10a, v10b]))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return a1, a2, 1.0
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, float(2 * norm.sf(abs(z)))


def wilcoxon_p(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        return 1.0
    return float(wilcoxon(a, b, zero_method="pratt").pvalue)


def mcnemar_exact(b, c):
    return 1.0 if b + c == 0 else float(binomtest(b, b + c, 0.5).pvalue)


def fmt(x, nd=3):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ metric tables
def full_metric_table(metrics, methods):
    metrics = metrics.copy()
    # Decision-tree leaf probabilities are 0/1, so logistic recalibration is not estimable.
    metrics.loc[metrics.classifier == "DecisionTree", ["cal_slope", "cal_intercept"]] = np.nan
    prevalence = 40 / 866
    metrics["oe_ratio"] = prevalence / metrics["mean_pred_prob"]
    rows = []
    for m in methods:
        for c in CLASSIFIERS:
            d = metrics[(metrics.method == m) & (metrics.classifier == c)]
            r = {"method": LABEL[m], "classifier": CLF_LABEL[c], "runs": len(d),
                 "n_synthetic_median": d["n_synthetic"].median()}
            for k in ALL_METRICS:
                q = d[k].quantile([0.25, 0.5, 0.75]).to_numpy()
                r[k] = "–" if np.isnan(q[1]) else f"{q[1]:.3f} [{q[0]:.3f}–{q[2]:.3f}]"
                r[k + "_median"] = q[1]
                r[k + "_min"], r[k + "_max"] = d[k].min(), d[k].max()
            rows.append(r)
    return pd.DataFrame(rows)


def threshold_table(methods):
    rows = []
    for m in methods:
        t = pd.read_csv(os.path.join(OUT, m, "thresholds.csv"))
        g = t.groupby(["classifier", "threshold"])[["sensitivity", "specificity", "ppv", "npv", "flagged_fraction"]].median()
        g = g.reset_index()
        g.insert(0, "method", LABEL[m])
        rows.append(g)
    out = pd.concat(rows, ignore_index=True)
    out["classifier"] = out["classifier"].map(CLF_LABEL)
    return out


# ------------------------------------------------------------------ uncertainty and paired tests
def bootstrap_table(methods, y):
    rows = []
    for m in methods:
        P = load_preds(m)
        for c in CLASSIFIERS:
            ci = bootstrap_ci(y, P[f"{c}__y_pred"].astype(int), P[f"{c}__score"])
            r = {"method": LABEL[m], "classifier": CLF_LABEL[c]}
            for k, (lo, hi) in ci.items():
                r[k + "_ci"] = f"[{lo:.3f}, {hi:.3f}]"
                r[k + "_lo"], r[k + "_hi"] = lo, hi
            rows.append(r)
        print(f"  bootstrap done: {m}", flush=True)
    return pd.DataFrame(rows)


def paired_tests(metrics, methods, y, ref):
    rows = []
    others = [m for m in methods if m != ref]
    preds = {m: load_preds(m) for m in methods}
    for c in CLASSIFIERS:
        # run-level Wilcoxon (paired by run index) with Holm correction per metric
        for k in KEY_METRICS:
            a = runs_matrix(metrics, ref, c, k)
            ps, diffs = [], []
            for o in others:
                b = runs_matrix(metrics, o, c, k)
                ps.append(wilcoxon_p(a, b)); diffs.append(np.median(a) - np.median(b))
            adj = multipletests(ps, method="holm")[1]
            try:
                fr = friedmanchisquare(*[runs_matrix(metrics, m, c, k) for m in methods]).pvalue
            except ValueError:
                fr = np.nan
            for o, p, pa, dlt in zip(others, ps, adj, diffs):
                rows.append({"classifier": CLF_LABEL[c], "metric": k, "test": "Wilcoxon signed-rank (25 paired runs)",
                             "comparison": f"{LABEL[ref]} vs {LABEL[o]}", "difference_in_medians": dlt,
                             "p_raw": p, "p_holm": pa, "friedman_p_all_methods": fr})
        # patient-level tests on the shared test set
        ref_score = preds[ref][f"{c}__score"].mean(0)
        ref_vote = (preds[ref][f"{c}__y_pred"].mean(0) > 0.5).astype(int)
        dl, mc_pos, mc_neg = [], [], []
        for o in others:
            o_score = preds[o][f"{c}__score"].mean(0)
            o_vote = (preds[o][f"{c}__y_pred"].mean(0) > 0.5).astype(int)
            a1, a2, p = delong_test(y, ref_score, o_score)
            dl.append((o, a1 - a2, p))
            pos = y == 1
            b = int(((ref_vote == 1) & (o_vote == 0) & pos).sum()); cc = int(((ref_vote == 0) & (o_vote == 1) & pos).sum())
            mc_pos.append((o, b - cc, mcnemar_exact(b, cc), b, cc))
            neg = y == 0
            b = int(((ref_vote == 0) & (o_vote == 1) & neg).sum()); cc = int(((ref_vote == 1) & (o_vote == 0) & neg).sum())
            mc_neg.append((o, b - cc, mcnemar_exact(b, cc), b, cc))
        for name, res, metric in [("DeLong (run-averaged scores)", dl, "auroc"),
                                  ("Exact McNemar, dialysis patients (majority vote over runs)", mc_pos, "recall"),
                                  ("Exact McNemar, non-dialysis patients (majority vote over runs)", mc_neg, "specificity")]:
            adj = multipletests([r[2] for r in res], method="holm")[1]
            for r, pa in zip(res, adj):
                row = {"classifier": CLF_LABEL[c], "metric": metric, "test": name,
                       "comparison": f"{LABEL[ref]} vs {LABEL[r[0]]}", "difference_in_medians": r[1],
                       "p_raw": r[2], "p_holm": pa, "friedman_p_all_methods": np.nan}
                if len(r) == 5:
                    row["discordant_ref_better"], row["discordant_other_better"] = r[3], r[4]
                rows.append(row)
    return pd.DataFrame(rows)


def original_saved_recalls():
    """Per-run class-1 recalls saved in notebook 01 (the values behind Table 3)."""
    saved = pd.read_csv(SAVED_RECALLS)
    return {m: {c: d[d.classifier == c].sort_values("run")["recall"].tolist() for c in CLASSIFIERS}
            for m, d in saved.groupby("method", sort=False)}


def original_ci_table(n_pos=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m, d in original_saved_recalls().items():
        for c in CLASSIFIERS:
            v = np.array(d[c])
            boots = [np.median(rng.choice(v, len(v))) for _ in range(B_BOOT)]
            med = np.median(v)
            lo_w, hi_w = proportion_confint(round(med * n_pos), n_pos, method="wilson")
            rows.append({"method": LABEL[m], "classifier": CLF_LABEL[c], "runs": len(v), "median": med,
                         "range": f"{v.min():.3f}–{v.max():.3f}", "mean": v.mean(), "sd": v.std(ddof=1),
                         "median_ci_over_runs": f"[{np.percentile(boots, 2.5):.3f}, {np.percentile(boots, 97.5):.3f}]",
                         "wilson_ci_test_set_40_dialysis": f"[{lo_w:.3f}, {hi_w:.3f}]"})
    return pd.DataFrame(rows)


def binomial_multiplicity(methods):
    df, *_ = load_split()
    df10 = df[df["dialysis"] == 1].drop(columns=["dialysis"])  # real dialysis records, as in the notebook
    feats = list(df10.columns)
    rows = []
    for m in methods:
        if m == "real":
            continue
        if m == "bgcs":
            cz, pz = np.load(os.path.join(OUT, m, "candidates.npz")), np.load(os.path.join(OUT, m, "gpt2_probs.npz"))
            sets = [cz[f"run{r}"][pz[f"run{r}"] <= 0.5] for r in range(len(pz.files))]
        else:
            z = np.load(os.path.join(OUT, m, "synthetic.npz"))
            sets = [z[k] for k in z.files if k.startswith("run")]
        for r, syn in enumerate(sets):
            syn = np.asarray(syn, float)
            p = []
            for j in range(len(feats)):
                c1, c2 = df10.iloc[:, j].sum(), syn[:, j].sum()
                with np.errstate(all="ignore"):
                    _, pv = proportions_ztest([c1, c2], [len(df10), len(syn)])
                p.append(1.0 if np.isnan(pv) else pv)
            p = np.array(p)
            rows.append({"method": LABEL[m], "run": r, "n_synthetic": len(syn),
                         "significant_raw_0.05": int((p < 0.05).sum()),
                         "significant_bonferroni_0.05": int(multipletests(p, 0.05, "bonferroni")[0].sum()),
                         "significant_bh_fdr_0.05": int(multipletests(p, 0.05, "fdr_bh")[0].sum()),
                         "mean_p": p.mean(), "features": len(p)})
    per_run = pd.DataFrame(rows)
    summ = per_run.groupby("method").agg(runs=("run", "count"), n_synthetic=("n_synthetic", "median"),
                                         sig_raw=("significant_raw_0.05", "median"),
                                         sig_bonferroni=("significant_bonferroni_0.05", "median"),
                                         sig_bh=("significant_bh_fdr_0.05", "median"),
                                         mean_p=("mean_p", "median")).reset_index()
    return per_run, summ


# ------------------------------------------------------------------ screening efficiency and ablation
def filter_efficiency():
    path = os.path.join(OUT, "bgcs", "filter_log.csv")
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path)
    f["acceptance_rate"] = f["kept_for_training"] / f["candidates"]
    cols = ["candidates", "predicted_label_1", "predicted_label_0", "kept_for_training", "acceptance_rate",
            "share_prob_gt_0.1", "share_prob_gt_0.3", "share_prob_gt_0.5", "share_prob_gt_0.7", "share_prob_gt_0.9",
            "copula_generation_s", "serialization_s", "gpt2_inference_s", "classifier_training_s"]
    return f[cols].agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "quantity"})


# ------------------------------------------------------------------ figures
def _axes_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e6e3", lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)


def fig_calibration(methods, y):
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 11)
    for ax, c in zip(axes.ravel(), CLASSIFIERS):
        ax.plot([0, 1], [0, 1], color="#9a9a96", lw=0.8, label="Perfect calibration")
        for m in methods:
            P = load_preds(m)[f"{c}__proba"]
            p = P.ravel(); yy = np.tile(y, P.shape[0])
            idx = np.minimum(np.digitize(p, bins) - 1, 9)
            xs, ys = [], []
            for b in range(10):
                sel = idx == b
                if sel.sum() >= 20:
                    xs.append(p[sel].mean()); ys.append(yy[sel].mean())
            col, ls, mk = STYLE[m]
            ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=4, lw=1.4, label=LABEL[m])
        ax.set_title(c.replace("LogisticRegression", "Logistic Regression").replace("DecisionTree", "Decision Tree")
                     .replace("RandomForest", "Random Forest"), fontsize=9)
        _axes_style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted probability", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed proportion with dialysis", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ["png", "tif"]:
        fig.savefig(os.path.join(AN, f"fig_calibration.{ext}"), dpi=300,
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
    plt.close(fig)


def fig_dca(methods, y):
    plt.rcParams["font.family"] = "Arial"
    prev = y.mean()
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.6), sharex=True, sharey=True)
    t = DCA_THRESHOLDS
    treat_all = prev - (1 - prev) * t / (1 - t)
    for ax, c in zip(axes.ravel(), CLASSIFIERS):
        ax.plot(t, treat_all, color="#9a9a96", lw=0.8, label="Treat all")
        ax.axhline(0, color="#52514e", lw=0.8, label="Treat none")
        for m in methods:
            nb = np.median(load_preds(m)[f"{c}__net_benefit"], axis=0)
            col, ls, _ = STYLE[m]
            ax.plot(t, nb, color=col, ls=ls, lw=1.4, label=LABEL[m])
        ax.set_ylim(-0.01, prev * 1.15)
        ax.set_title(c.replace("LogisticRegression", "Logistic Regression").replace("DecisionTree", "Decision Tree")
                     .replace("RandomForest", "Random Forest"), fontsize=9)
        _axes_style(ax)
    for ax in axes[1]:
        ax.set_xlabel("Threshold probability", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Net benefit", fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ["png", "tif"]:
        fig.savefig(os.path.join(AN, f"fig_decision_curves.{ext}"), dpi=300,
                    **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    os.makedirs(AN, exist_ok=True)
    methods = available()
    ref = "bgcs" if "bgcs" in methods else "binary_copula"
    print("methods:", methods, "| reference for paired tests:", ref, flush=True)
    _, _, _, _, _, y_test = load_split()
    y = y_test.to_numpy().astype(int)
    metrics = load_metrics(methods)

    full = full_metric_table(metrics, methods)
    full.to_csv(os.path.join(AN, "table_full_metrics.csv"), index=False)
    threshold_table(methods).to_csv(os.path.join(AN, "table_thresholds.csv"), index=False)
    fig_calibration(methods, y)
    fig_dca(methods, y)
    print("metric tables and figures done", flush=True)

    boot = bootstrap_table(methods, y)
    boot.to_csv(os.path.join(AN, "table_bootstrap_ci.csv"), index=False)
    tests = paired_tests(metrics, methods, y, ref)
    tests.to_csv(os.path.join(AN, "table_paired_tests.csv"), index=False)
    orig = original_ci_table()
    orig.to_csv(os.path.join(AN, "table_original_recall_ci.csv"), index=False)
    per_run, binom = binomial_multiplicity(methods)
    per_run.to_csv(os.path.join(AN, "table_binomial_multiplicity_per_run.csv"), index=False)
    binom.to_csv(os.path.join(AN, "table_binomial_multiplicity.csv"), index=False)
    print("uncertainty and test tables done", flush=True)

    abl = full[full.method.isin([LABEL["gc"], LABEL["binary_copula"], LABEL["bgcs"]])]
    abl[["method", "classifier", "n_synthetic_median"] + [k for k in KEY_METRICS]].to_csv(
        os.path.join(AN, "table_ablation.csv"), index=False)
    eff = filter_efficiency()
    if eff is not None:
        eff.to_csv(os.path.join(AN, "table_filter_efficiency.csv"), index=False)
    print("all analysis outputs written to", AN, flush=True)


if __name__ == "__main__":
    main()

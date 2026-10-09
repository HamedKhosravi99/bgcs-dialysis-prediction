"""Shared setup: data loading, the 80/20 split, the four classifiers and the evaluation metrics.

Reproduces the data, split and classifiers of the two notebooks in notebooks/ and adds
predicted probabilities, a full metric set, calibration, threshold analysis and timings.

Paths can be changed with environment variables:
  BGCS_DATA  the data file (default: data/merged_90_c1.xlsx)
  BGCS_OUT   the output folder (default: outputs/)
"""
import os
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("BGCS_DATA", os.path.join(ROOT, "data", "merged_90_c1.xlsx"))
OUT = os.environ.get("BGCS_OUT", os.path.join(ROOT, "outputs"))
CLASSIFIERS = ["DecisionTree", "RandomForest", "SVM", "LogisticRegression"]
N_RUNS = 25
OP_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DCA_THRESHOLDS = np.round(np.arange(0.01, 0.61, 0.01), 2)


_DF = None


def load_split():
    """Same data and 80/20 split (random_state=42) as the notebooks."""
    global _DF
    if _DF is None:
        if not os.path.exists(DATA):
            raise FileNotFoundError(f"Data file not found: {DATA}. See data/README.md.")
        _DF = pd.read_excel(DATA)
    df = _DF.drop(["patient_id"], axis=1)
    X = df.drop("dialysis", axis=1)
    y = df["dialysis"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    return df, train_df, X_train, X_test, y_train, y_test


def make_classifiers(lr_kwargs=None, rf_jobs=4):
    """Default classifiers exactly as in the notebooks.

    SVC gets probability=True only so calibration and decision curves can be computed;
    its predict() output is unchanged (checked on the binary-copula data).
    """
    lr_kwargs = {"random_state": 42} if lr_kwargs is None else lr_kwargs
    return {
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_jobs=rf_jobs),
        "SVM": SVC(random_state=42, probability=True),
        "LogisticRegression": LogisticRegression(**lr_kwargs),
    }


def predict_all(clf, name, X_test):
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    score = clf.decision_function(X_test) if name == "SVM" else proba
    return y_pred.astype(np.int8), proba.astype(np.float64), score.astype(np.float64)


def calibration(y, p, n_bins=10):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(pc / (1 - pc))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
    except Exception:
        slope = np.nan
    try:
        citl = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit().params[0]
    except Exception:
        citl = np.nan
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += abs(p[m].mean() - y[m].mean()) * m.mean()
    return {"cal_slope": slope, "cal_intercept": citl, "ece": ece, "mean_pred_prob": float(p.mean())}


def metrics_from(y, y_pred, proba, score):
    y = np.asarray(y).astype(int)
    tp = int(((y_pred == 1) & (y == 1)).sum())
    fn = int(((y_pred == 0) & (y == 1)).sum())
    fp = int(((y_pred == 1) & (y == 0)).sum())
    tn = int(((y_pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    den = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    m = {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "recall": sens, "specificity": spec, "precision": ppv, "npv": npv, "f1": f1,
        "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2,
        "mcc": (tp * tn - fp * fn) / den if den else 0.0,
        "auroc": roc_auc_score(y, score), "auprc": average_precision_score(y, score),
        "brier": brier_score_loss(y, proba),
    }
    m.update(calibration(y, proba))
    return m


def threshold_rows(y, p):
    y = np.asarray(y).astype(int)
    rows = []
    for t in OP_THRESHOLDS:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
        rows.append({
            "threshold": t, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "flagged_fraction": (tp + fp) / len(y), "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return rows


def net_benefit(y, p):
    y = np.asarray(y).astype(int)
    n = len(y)
    out = []
    for t in DCA_THRESHOLDS:
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum()
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


class RunRecorder:
    """Collects per-run metrics, threshold tables, predictions and timings for one method."""

    def __init__(self, method, y_test):
        self.method = method
        self.dir = os.path.join(OUT, method)
        os.makedirs(self.dir, exist_ok=True)
        self.y_test = np.asarray(y_test).astype(np.int8)
        self.metric_rows, self.thr_rows, self.extra_rows = [], [], []
        self.preds = {c: {"y_pred": [], "proba": [], "score": [], "net_benefit": []} for c in CLASSIFIERS}

    def fit_and_record(self, run, X_train, y_train, X_test, lr_kwargs=None, n_synthetic=None, extra=None):
        import time
        for name, clf in make_classifiers(lr_kwargs).items():
            t0 = time.time()
            clf.fit(X_train, y_train)
            fit_s = time.time() - t0
            y_pred, proba, score = predict_all(clf, name, X_test)
            m = metrics_from(self.y_test, y_pred, proba, score)
            m.update({"method": self.method, "run": run, "classifier": name, "n_train": len(y_train),
                      "n_synthetic": n_synthetic, "fit_seconds": fit_s})
            if extra:
                m.update(extra)
            self.metric_rows.append(m)
            for r in threshold_rows(self.y_test, proba):
                r.update({"method": self.method, "run": run, "classifier": name})
                self.thr_rows.append(r)
            p = self.preds[name]
            p["y_pred"].append(y_pred); p["proba"].append(proba); p["score"].append(score)
            p["net_benefit"].append(net_benefit(self.y_test, proba))

    def save(self):
        pd.DataFrame(self.metric_rows).to_csv(os.path.join(self.dir, "metrics.csv"), index=False)
        pd.DataFrame(self.thr_rows).to_csv(os.path.join(self.dir, "thresholds.csv"), index=False)
        arrays = {"y_test": self.y_test, "dca_thresholds": DCA_THRESHOLDS}
        for c, d in self.preds.items():
            for k, v in d.items():
                if v:
                    arrays[f"{c}__{k}"] = np.vstack(v)
        np.savez_compressed(os.path.join(self.dir, "predictions.npz"), **arrays)

"""Fig 6: the truncated CDSS decision tree, drawn with readable fonts.

Rebuilds the tree of notebook 01 cells 202/207/213-219: training data plus the last
(25th) binary-copula synthetic set (np.random.seed(0), 2,407 rows), DecisionTreeClassifier with default
settings, drawn to depth 3 as in the paper. The notebook did not fix a seed, so random_state=0 is used,
which reproduces the published tree. The script checks this against the published figure's top nodes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import OUT, load_split
from run_methods import generate_gaussian_copula

# Nodes visible in the published Fig 6 (feature, [non-dialysis, dialysis]) in breadth-first order
PUBLISHED = [("code_G0480", [3261, 2608]), ("code_Z09", [400, 1835]), ("code_85025", [2861, 773]),
             ("code_M79", [361, 809]), ("code_Z79.899", [39, 1026]), ("code_36415", [33, 236]),
             ("code_V45.89", [2828, 537])]


def top_nodes(tree, names, k=7):
    t = tree.tree_
    order, queue = [], [0]
    while queue and len(order) < k:
        n = queue.pop(0)
        if t.children_left[n] == -1:
            continue
        counts = np.round(t.value[n][0] * t.weighted_n_node_samples[n]).astype(int).tolist()
        order.append((names[t.feature[n]], counts))
        queue += [t.children_left[n], t.children_right[n]]
    return order


def draw_tree(clf, names, fig_dir, max_depth=3):
    """Compact, PLOS-sized (7.5 in wide, 8-pt Arial) drawing of the top levels of the tree."""
    from matplotlib.patches import FancyBboxPatch
    t = clf.tree_
    nodes = {}  # node id -> (depth, x)
    def walk(n, depth, lo, hi):
        nodes[n] = (depth, (lo + hi) / 2)
        if depth < max_depth and t.children_left[n] != -1:
            mid = (lo + hi) / 2
            walk(t.children_left[n], depth + 1, lo, mid)
            walk(t.children_right[n], depth + 1, mid, hi)
    walk(0, 0, 0.0, 1.0)
    plt.rcParams["font.family"] = "Arial"
    W, H = 7.5, 4.9
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    level_y = [H - 0.55, H - 1.65, H - 2.75, H - 3.85]
    bw, bh = 0.86, 0.72
    def xy(n):
        d, x = nodes[n]
        return 0.1 + x * (W - 0.2), level_y[d]
    for n, (d, _) in nodes.items():
        x, y = xy(n)
        counts = np.round(t.value[n][0] * t.weighted_n_node_samples[n]).astype(int)
        p = counts / counts.sum()
        k = int(np.argmax(p))
        alpha = (p.max() - p.min()) / (1 - p.min()) if p.min() < 1 else 1.0
        base = np.array([229, 129, 57]) / 255 if k == 0 else np.array([57, 157, 229]) / 255
        color = 1 - alpha * (1 - base)
        lines = []
        if t.children_left[n] != -1:
            lines.append(f"{names[t.feature[n]]} \u2264 0.5")
        lines += [f"Gini = {t.impurity[n]:.3f}", f"n = {int(t.n_node_samples[n]):,}",
                  f"[{counts[0]:,} / {counts[1]:,}]", ["Non-dialysis", "Dialysis"][k]]
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc=color, ec="0.35", lw=0.6))
        ax.text(x, y, "\n".join(lines), ha="center", va="center", fontsize=8, linespacing=1.15)
        if t.children_left[n] != -1:
            for child, label in [(t.children_left[n], "absent"), (t.children_right[n], "present")]:
                if child in nodes:
                    cx, cy = xy(child)
                    ax.annotate("", xy=(cx, cy + bh / 2), xytext=(x, y - bh / 2),
                                arrowprops=dict(arrowstyle="-|>", lw=0.6, color="0.3", shrinkA=0, shrinkB=0))
                    ax.text((x + cx) / 2 + (-0.12 if label == "absent" else 0.12), (y - bh / 2 + cy + bh / 2) / 2,
                            label, ha="right" if label == "absent" else "left", va="center", fontsize=8,
                            style="italic", color="0.25")
                else:  # subtree truncated for display
                    sx = x + (-0.22 if label == "absent" else 0.22)
                    sy = y - bh / 2 - 0.42
                    ax.plot([x, sx], [y - bh / 2, sy + 0.1], color="0.5", lw=0.6, ls=(0, (2, 2)))
                    ax.add_patch(FancyBboxPatch((sx - 0.15, sy - 0.1), 0.3, 0.2, boxstyle="round,pad=0.01",
                                                fc="0.85", ec="0.5", lw=0.5))
                    ax.text(sx, sy, "(...)", ha="center", va="center", fontsize=8, color="0.25")
    png = os.path.join(fig_dir, "Fig6_tree_redrawn.png")
    fig.savefig(png, dpi=300)
    plt.close(fig)
    # PLOS figure files are RGB TIFF without an alpha channel
    Image.open(png).convert("RGB").save(os.path.join(fig_dir, "Fig6_tree_redrawn.tif"), dpi=(300, 300),
                                        compression="tiff_lzw")


def build_cdss_tree():
    """The CDSS tree of notebook 01: training data plus the 25th binary-copula synthetic set."""
    _, train_df, X_train, *_ = load_split()
    np.random.seed(0)
    minority_data = train_df[train_df["dialysis"] == 1].drop(columns=["dialysis"])
    marginal_probs = minority_data.mean().values
    cor_matrix = minority_data.corr().values
    for _ in range(25):
        syn = generate_gaussian_copula(cor_matrix, 2407, marginal_probs)
    generated_df = pd.DataFrame(syn, columns=minority_data.columns)
    generated_df["dialysis"] = 1
    balanced_df = pd.concat([train_df, generated_df], ignore_index=True)
    X = balanced_df.drop("dialysis", axis=1)
    y = balanced_df["dialysis"]

    # Notebook 01 did not fix a seed for this tree, and the tree breaks ties between equally good splits at random.
    # random_state=0 reproduces the published Fig 6, and the check below confirms the match.
    clf = DecisionTreeClassifier(random_state=0).fit(X, y)
    return clf, X


def main():
    clf, X = build_cdss_tree()
    seed_used = 0
    match = top_nodes(clf, X.columns) == PUBLISHED
    print("matches published Fig 6 top nodes:", match, "| random_state:", seed_used,
          "| depth:", clf.get_depth(), "| leaves:", clf.get_n_leaves())

    fig_dir = os.path.join(OUT, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    draw_tree(clf, [c.replace("code_", "") for c in X.columns], fig_dir)
    with open(os.path.join(fig_dir, "Fig6_tree_info.txt"), "w") as f:
        f.write(f"matches_published_top_nodes={match}\nrandom_state={seed_used}\n"
                f"full_tree_depth={clf.get_depth()}\nfull_tree_leaves={clf.get_n_leaves()}\n")


if __name__ == "__main__":
    main()

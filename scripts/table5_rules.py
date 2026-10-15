"""Table 5: the decision rules of the CDSS tree and the codes whose splits separate dialysis patients.

Notebook 01 cell 220 printed the rules of the CDSS tree, and the Table 5 codes were chosen by reading them.
They are clinically relevant codes whose splits send patients into branches made up almost entirely of
dialysis cases. This script prints the same rules for the rebuilt tree (fig6_tree.build_cdss_tree) and lists,
for every code, its splits and the largest branch that contains only dialysis patients.

The notebook did not fix the tree's random seed. The rebuilt tree matches the published Fig 6 and the notebook's
feature importances, but a few deep splits differ (383 leaves here vs 385 in the notebook).

The outputs contain small patient counts at deep nodes. They go to outputs/table5/ and must not be shared.
"""
import os

import numpy as np
import pandas as pd

from common import OUT
from fig6_tree import build_cdss_tree

TABLE5 = ["E87.5", "E87.6", "I25.2", "I10", "81001", "84100", "I11.0", "84295", "I51.7", "R94.31"]


def counts(t, node):
    """[non-dialysis, dialysis] patients at a node (scikit-learn stores fractions in recent versions)."""
    v = t.value[node][0]
    if v.sum() <= 1.0 + 1e-9:
        v = v * t.weighted_n_node_samples[node]
    return [int(round(x)) for x in v]


def main():
    clf, X = build_cdss_tree()
    t = clf.tree_
    names = [c.replace("code_", "").strip() for c in X.columns]
    out_dir = os.path.join(OUT, "table5")
    os.makedirs(out_dir, exist_ok=True)

    lines, rows = [], []

    def recurse(node, depth):
        indent = "  " * depth
        if t.children_left[node] == -1:
            lines.append(f"{indent}then: {counts(t, node)}")
            return
        f = names[t.feature[node]]
        left, right = t.children_left[node], t.children_right[node]
        rows.append({"code": f, "depth": depth, "node": counts(t, node),
                     "absent": counts(t, left), "present": counts(t, right)})
        lines.append(f"{indent}if {f} absent:")
        recurse(left, depth + 1)
        lines.append(f"{indent}else:  # {f} present")
        recurse(right, depth + 1)

    recurse(0, 0)
    with open(os.path.join(out_dir, "tree_rules.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    splits = pd.DataFrame(rows)
    splits["all_dialysis_branch"] = [max([b[1] for b in (r.absent, r.present) if b[0] == 0] or [0])
                                     for r in splits.itertuples()]
    splits.to_csv(os.path.join(out_dir, "splits.csv"), index=False)
    best = splits.groupby("code")["all_dialysis_branch"].max().sort_values(ascending=False)
    best.to_csv(os.path.join(out_dir, "largest_all_dialysis_branch_per_code.csv"))

    print(f"tree: depth {clf.get_depth()}, {clf.get_n_leaves()} leaves")
    print(f"codes with at least one all-dialysis branch: {(best > 0).sum()} of {len(names)}")
    for code in TABLE5:
        s = splits[splits.code == code]
        print(f"  {code:7s} splits {len(s)} | largest all-dialysis branch {int(best.get(code, 0))}")


if __name__ == "__main__":
    main()

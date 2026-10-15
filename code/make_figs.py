import json, sys, os
import numpy as np
from figstyle import *
from common import WORK, CLASS_NAMES

FIG = os.path.join(os.path.dirname(__file__), "..", "figures")
J = lambda n: json.load(open(f"{WORK}/{n}"))


def tuning():
    A = J("rf_stageA.json")
    tree_ord = max(J("tree_encoding.json"), key=lambda r: r["macro_f1"])["macro_f1"]
    qs = [1, 2, 4, 6, 10, 16, 24, 42]
    W = [("None", "Uniform weights", "o", "-"),
         ("balanced", "Weights from training partition", "s", "-"),
         ("balanced_subsample", "Weights per bootstrap sample", "^", "--")]

    f0, a0 = plt.subplots(figsize=(COL, 2.3)); f1, a1 = plt.subplots(figsize=(COL, 2.3))
    ax = [a0, a1]
    for (w, lab, mk, ls), col in zip(W, ["#0072B2", "#CC79A7", "#009E73"]):
        r = sorted([x for x in A if x["class_weight"] == w], key=lambda x: x["max_features"])
        ax[0].plot(qs, [x["macro_f1"] for x in r], marker=mk, ls=ls, color=col, label=lab)
    ax[0].axhline(tree_ord, color=C["tree"], ls=":", lw=1, label="Tuned classification tree")
    ax[0].set_xscale("log"); ax[0].set_xticks(qs); ax[0].set_xticklabels(qs)
    ax[0].minorticks_off()
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("Features per split, $q$")
    ax[0].set_ylabel(r"Mean $\bar{F}$ over folds")
    ax[0].legend(loc="lower right", frameon=False)
    r = sorted([x for x in A if x["class_weight"] == "balanced"], key=lambda x: x["max_features"])
    ax[1].plot(qs, [x["macro_f1"] for x in r], marker="s", color="#CC79A7", label=r"Forest $\bar{F}$")
    ax[1].plot(qs, [x["strength"] for x in r], marker="o", color=C["grey"], label=r"Mean $\bar{F}$ of single members")
    ax[1].plot(qs, [x["agreement"] for x in r], marker="D", color="#E69F00", label="Mean pairwise agreement")
    ax[1].set_xscale("log"); ax[1].set_xticks(qs); ax[1].set_xticklabels(qs); ax[1].minorticks_off()
    ax[1].set_ylim(0, 1); ax[1].set_xlabel("Features per split, $q$"); ax[1].set_ylabel("Value")
    ax[1].legend(loc="lower right", frameon=False)
    f0.savefig(f"{FIG}/rf_q_a.pdf"); f1.savefig(f"{FIG}/rf_q_b.pdf"); plt.close("all")

    B = J("rf_stageB.json")
    f0, a0 = plt.subplots(figsize=(COL, 2.1)); f1, a1 = plt.subplots(figsize=(COL, 2.1))
    ax = [a0, a1]
    leaf = sorted([x for x in B if "min_samples_leaf" in x], key=lambda x: x["min_samples_leaf"])
    dep = sorted([x for x in B if "max_depth" in x], key=lambda x: x["max_depth"])
    full = [x for x in leaf if x["min_samples_leaf"] == 1][0]
    for a, rows, key, xl in [(ax[0], leaf, "min_samples_leaf", "Minimum leaf size"),
                             (ax[1], dep, "max_depth", "Maximum depth")]:
        xs = [x[key] for x in rows]
        a.plot(xs, [x["train_macro_f1"] for x in rows], marker="o", ls="--", color=C["grey"], label="Training partitions")
        a.plot(xs, [x["macro_f1"] for x in rows], marker="s", color="#CC79A7", label="Test partitions")
        if key == "max_depth":
            a.axhline(full["macro_f1"], color="#CC79A7", ls=":", lw=1, label="Unlimited depth, test")
        a.set_ylim(0, 1); a.set_xlabel(xl); a.set_ylabel(r"Mean $\bar{F}$ over folds")
        a.legend(loc="lower right", frameon=False)
    ax[0].set_xscale("log"); ax[0].set_xticks([1, 2, 5, 10, 20, 50]); ax[0].set_xticklabels([1, 2, 5, 10, 20, 50]); ax[0].minorticks_off()
    ax[1].set_xticks([8, 12, 16, 20, 24, 32])
    f0.savefig(f"{FIG}/rf_leaf.pdf"); f1.savefig(f"{FIG}/rf_depth.pdf"); plt.close("all")

    try:
        cur = J("rf_curve.json")
    except Exception:
        return
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    ins = ax.inset_axes([0.42, 0.12, 0.55, 0.42])
    for a in (ax, ins):
        for k, lab, col in [("best", "Random forest", "#CC79A7"), ("bagging", "Bagged trees", C["bagging"])]:
            m = np.array(cur[k]["mean"]); s_ = np.array(cur[k]["std"]); t = np.arange(1, len(m) + 1)
            a.plot(t, m, color=col, label=lab, lw=1.2)
            a.fill_between(t, m - s_, m + s_, color=col, alpha=0.15, lw=0)
        a.axhline(tree_ord, color=C["tree"], ls=(0, (2, 1.5)), lw=1, label="Tuned classification tree")
        a.axvline(100, color=C["grey"], ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(1, 300); ax.set_ylim(0, 1)
    ax.set_xticks([1, 10, 100, 300]); ax.set_xticklabels([1, 10, 100, 300]); ax.minorticks_off()
    ax.set_xlabel(r"Number of trees, $\tau$"); ax.set_ylabel(r"Mean $\bar{F}$ over folds")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    ins.set_xlim(10, 300); ins.set_ylim(0.56, 0.61); ins.set_xscale("log")
    ins.set_xticks([10, 100, 300]); ins.set_xticklabels([10, 100, 300]); ins.minorticks_off()
    ins.tick_params(labelsize=7); ins.set_facecolor("white")
    for sp in ins.spines.values(): sp.set_visible(True); sp.set_linewidth(0.6)
    fig.savefig(f"{FIG}/rf_trees.pdf"); plt.close(fig)


def final():
    F = J("final.json"); orc = J("oracle.json")
    models = ["knn", "tree", "bagging", "rf"]
    f1 = {"knn": np.array(F["knn"]["f1"])}
    for m in ["tree", "bagging", "rf"]:
        f1[m] = np.mean([r["f1"] for r in F[m]["runs"]], axis=0)
    n = np.bincount(np.load(f"{WORK}/test_preds.npz")["y"], minlength=10)
    order = np.argsort(-n)
    fig, ax = plt.subplots(figsize=(FULL, 2.4))
    w = 0.2; x = np.arange(10)
    for i, m in enumerate(models):
        ax.bar(x + (i - 1.5) * w, f1[m][order], w, color=C[m], label=LABEL[m])
    short = ["Normal", "Recon.", "Backdoor", "DoS", "Exploits", "Analysis",
             "Fuzzers", "Worms", "Shellcode", "Generic"]
    ax.set_xticks(x); ax.set_xticklabels([f"{short[i]}\n({n[i]})" for i in order], fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel(r"$F_j$ on generalisation set")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    ax.grid(axis="x", visible=False)
    fig.savefig(f"{FIG}/per_class.pdf"); plt.close(fig)

    cm = np.array(F["rf_confusion"], float); cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(COL, 3.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    short = ["Normal", "Recon.", "Backdoor", "DoS", "Exploits", "Analysis",
             "Fuzzers", "Worms", "Shellcode", "Generic"]
    ax.set_xticks(range(10)); ax.set_xticklabels(short, rotation=90, fontsize=8)
    ax.set_yticks(range(10)); ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted category"); ax.set_ylabel("True category"); ax.grid(False)
    for i in range(10):
        for j in range(10):
            if cm[i, j] >= 0.005:
                ax.text(j, i, f"{100 * cm[i, j]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if cm[i, j] > 0.5 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=8)
    cb.set_label("Proportion of true category", fontsize=8)
    fig.savefig(f"{FIG}/rf_confusion.pdf"); plt.close(fig)


if __name__ == "__main__":
    {"tuning": tuning, "final": final}[sys.argv[1] if len(sys.argv) > 1 else "tuning"]()

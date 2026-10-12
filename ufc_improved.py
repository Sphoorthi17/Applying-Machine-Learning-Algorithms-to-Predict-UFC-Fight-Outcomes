"""
UFC fight prediction - IMPROVED pipeline (extension of the paper reproduction)

Three improvements over ufc_prediction.py:
  1. Difference features : replace every R_x / B_x pair with (R_x - B_x). Halves the feature
                           count and makes the model about the *matchup*, not the corner.
  2. Swap augmentation   : for every TRAINING fight add a mirrored copy (Red <-> Blue, so the
                           diff features are negated and the label is flipped). Removes the
                           Red-corner bias and doubles the training data. Test folds are
                           never augmented.
  3. Regularization      : L2/L1 logistic regression, L2-penalised MLP, and
                           shallow/subsampled tree ensembles instead of unconstrained models.

Same time-series 20-fold evaluation as the paper reproduction, so numbers are comparable.
Keep this file in the same folder as ufc_prediction.py (it imports from it).

Usage:
    python ufc_improved.py --data data\\data.csv
    python ufc_improved.py --data data\\data.csv --ablation        # also run "diff only" (no swap)
    python ufc_improved.py --data data\\data.csv --skip-baseline   # skip the slow paper GB run
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ufc_prediction import RANDOM_STATE, get_models, load_data

FLIP = {"Red": "Blue", "Blue": "Red", "Draw": "Draw"}
LABELS = ["Red", "Blue", "Draw"]


# --------------------------------------------------------------------------
# 1. Difference features
# --------------------------------------------------------------------------
def make_diff_features(X):
    """(R_x - B_x) for every stat that exists for either corner."""
    suffixes = sorted({c[2:] for c in X.columns if c[:2] in ("R_", "B_")})
    ignored = [c for c in X.columns if c[:2] not in ("R_", "B_")]
    zero = pd.Series(0.0, index=X.index)
    out = {f"diff_{s}": X.get("R_" + s, zero) - X.get("B_" + s, zero) for s in suffixes}
    Xd = pd.DataFrame(out, index=X.index)
    print(f"Difference features: {X.shape[1]} raw -> {Xd.shape[1]} diff features "
          f"({len(ignored)} unpaired columns ignored)\n")
    return Xd


# --------------------------------------------------------------------------
# 3. Regularized models
# --------------------------------------------------------------------------
def get_improved_models():
    return {
        "LogReg L2 (C=0.01)": LogisticRegression(C=0.01, max_iter=2000,
                                                 random_state=RANDOM_STATE),
        "LogReg L1 (C=0.05)": LogisticRegression(penalty="l1", C=0.05, solver="saga", max_iter=3000, tol=1e-3,
                                                 random_state=RANDOM_STATE),
        "MLP (L2 alpha=1, 8 units)": MLPClassifier(hidden_layer_sizes=(8,), alpha=1.0,
                                                   solver="lbfgs", max_iter=500,
                                                   random_state=RANDOM_STATE),
        "Random Forest (min_leaf=10)": RandomForestClassifier(n_estimators=300,
                                                              min_samples_leaf=10,
                                                              max_features="sqrt", n_jobs=-1,
                                                              random_state=RANDOM_STATE),
        "Gradient Boosting (reg)": GradientBoostingClassifier(n_estimators=200,
                                                              learning_rate=0.03,
                                                              max_depth=2, subsample=0.7,
                                                              min_samples_leaf=20,
                                                              max_features="sqrt",
                                                              random_state=RANDOM_STATE),
    }


# --------------------------------------------------------------------------
# 2. Time-series CV with optional swap augmentation (training folds only)
# --------------------------------------------------------------------------
def run_cv(models, X, y, n_splits, augment, verbose=True):
    rows, all_true = [], []
    all_pred = {n: [] for n in models}
    for fold, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X), 1):
        Xtr, ytr = X.iloc[tr].values, y.iloc[tr]
        Xte, yte = X.iloc[te].values, y.iloc[te]
        if augment:                                   # mirrored copy: negate diffs, flip label
            Xtr = np.vstack([Xtr, -Xtr])
            ytr = pd.concat([ytr, ytr.map(FLIP)], ignore_index=True)
        sc = StandardScaler().fit(Xtr)                # scaler sees training fold only
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        base = (yte == "Red").mean()
        all_true.extend(yte.tolist())
        for name, model in models.items():
            model.fit(Xtr_s, ytr)
            ptr, pte = model.predict(Xtr_s), model.predict(Xte_s)
            all_pred[name].extend(pte.tolist())
            rows.append(dict(fold=fold, model=name,
                             train_acc=accuracy_score(ytr, ptr),
                             test_acc=accuracy_score(yte, pte),
                             test_f1_macro=f1_score(yte, pte, average="macro"),
                             baseline_red=base))
        if verbose:
            print(f"  fold {fold:2d}/{n_splits} done (train={len(Xtr)}, test={len(te)})")
    return pd.DataFrame(rows), all_pred, all_true


def summarize(res):
    res = res.assign(beats_red=res["test_acc"] > res["baseline_red"])
    s = res.groupby("model").agg(avg_train_acc=("train_acc", "mean"),
                                 avg_test_acc=("test_acc", "mean"),
                                 std_test_acc=("test_acc", "std"),
                                 avg_test_f1_macro=("test_f1_macro", "mean"),
                                 folds_beating_red=("beats_red", "sum"))
    n_folds = res["fold"].nunique()
    s["folds_beating_red"] = s["folds_beating_red"].astype(int).astype(str) + f"/{n_folds}"
    s = s.sort_values("avg_test_acc", ascending=False).round(4)
    red = res.groupby("fold")["baseline_red"].first()
    s.loc["Always guess Red"] = [np.nan, red.mean(), red.std(), np.nan, "-"]
    return s.round(4)


# --------------------------------------------------------------------------
# Regularization sweep + L1 feature selection
# --------------------------------------------------------------------------
def regularization_sweep(Xd, y, n_splits, out_dir):
    Cs = [1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0, 10.0]
    rows = []
    for C in Cs:
        models = {"L2": LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE),
                  "L1": LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000, tol=1e-3,
                                           random_state=RANDOM_STATE)}
        res, _, _ = run_cv(models, Xd, y, n_splits, augment=True, verbose=False)
        for name, g in res.groupby("model"):
            rows.append(dict(C=C, penalty=name, train_acc=g.train_acc.mean(),
                             test_acc=g.test_acc.mean(), baseline=g.baseline_red.mean()))
        print(f"  sweep C={C:g} done")
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(out_dir, "regularization_sweep.csv"), index=False)

    plt.figure(figsize=(8, 5))
    for pen, g in sw.groupby("penalty"):
        plt.plot(g.C, g.test_acc, marker="o", label=f"{pen} test")
        plt.plot(g.C, g.train_acc, marker="x", ls="--", label=f"{pen} train")
    plt.axhline(sw.baseline.iloc[0], color="k", ls=":", label="Always guess Red")
    plt.xscale("log"); plt.xlabel("C (smaller = stronger regularization)")
    plt.ylabel("Accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Logistic regression: effect of regularization strength")
    plt.savefig(os.path.join(out_dir, "regularization_sweep.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  (sweep is diagnostic: picking the best C from these same folds is optimistic)")


def l1_selected_features(Xd, y, out_dir, C=0.05):
    """Which diff features survive L1 shrinkage? (fit on all data - interpretation only)"""
    Xa = np.vstack([Xd.values, -Xd.values])
    ya = pd.concat([y, y.map(FLIP)], ignore_index=True)
    sc = StandardScaler().fit(Xa)
    m = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000, tol=1e-3,
                           random_state=RANDOM_STATE).fit(sc.transform(Xa), ya)
    coef = pd.Series(m.coef_[list(m.classes_).index("Red")], index=Xd.columns)
    kept = coef[coef != 0].sort_values(key=abs, ascending=False)
    kept.to_csv(os.path.join(out_dir, "l1_selected_features.csv"), header=["coef_for_Red"])
    print(f"\nL1 (C={C}) keeps {len(kept)}/{len(coef)} features. Top 15 by |coef|:")
    print(kept.head(15).round(4).to_string())


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def fig_fold_accuracy(res, out_dir):
    plt.figure(figsize=(10, 5.5))
    for name, g in res.groupby("model"):
        plt.plot(g["fold"], g["test_acc"], marker="o", ms=3, label=name)
    red = res.groupby("fold")["baseline_red"].first()
    plt.plot(red.index, red.values, "k--", label="Always guess Red")
    plt.xlabel("Fold"); plt.ylabel("Test accuracy"); plt.grid(alpha=.3)
    plt.legend(fontsize=7); plt.title("Test accuracy per time-series fold (improved pipeline)")
    plt.savefig(os.path.join(out_dir, "fold_accuracy_improved.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def fig_confusion(all_pred, all_true, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for ax, (name, pred) in zip(axes, all_pred.items()):
        ConfusionMatrixDisplay.from_predictions(all_true, pred, labels=LABELS,
                                                normalize="true", values_format=".2f",
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=9)
    for ax in axes[len(all_pred):]:
        ax.axis("off")
    plt.suptitle("Confusion matrices - improved pipeline (row-normalised, pooled over folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices_improved.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--out", default="results/improved")
    ap.add_argument("--ablation", action="store_true", help="also run diff features WITHOUT swap augmentation")
    ap.add_argument("--skip-baseline", action="store_true", help="skip the paper's Gradient Boosting run")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, _ = load_data(args.data)
    Xd = make_diff_features(X)

    print("== Improved pipeline: diff features + swap augmentation + regularized models ==")
    res, all_pred, all_true = run_cv(get_improved_models(), Xd, y, args.folds, augment=True)
    res["model"] += " [diff+swap]"
    all_pred = {k + " [diff+swap]": v for k, v in all_pred.items()}
    parts = [res]

    if args.ablation:
        print("\n== Ablation: diff features only (no swap augmentation) ==")
        r2, _, _ = run_cv(get_improved_models(), Xd, y, args.folds, augment=False)
        r2["model"] += " [diff only]"
        parts.append(r2)

    if not args.skip_baseline:
        print("\n== Reference: paper's Gradient Boosting on original features ==")
        gb = {"Paper GB [original features]": get_models()["Gradient Boosting"]}
        r3, _, _ = run_cv(gb, X, y, args.folds, augment=False)
        parts.append(r3)

    allres = pd.concat(parts, ignore_index=True)
    allres.to_csv(os.path.join(args.out, "fold_results_improved.csv"), index=False)
    summary = summarize(allres)
    summary.to_csv(os.path.join(args.out, "summary_improved.csv"))
    pd.set_option("display.width", 200)
    print("\n=== Average results over folds ===")
    print(summary.to_string())

    fig_fold_accuracy(res, args.out)
    fig_confusion(all_pred, all_true, args.out)
    print("\n== Regularization sweep (logistic regression) ==")
    regularization_sweep(Xd, y, args.folds, args.out)
    l1_selected_features(Xd, y, args.out)
    print(f"\nAll outputs saved in '{args.out}/'")


if __name__ == "__main__":
    main()

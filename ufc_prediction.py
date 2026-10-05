"""
UFC Fight Outcome Prediction (reproduction of McQuaide, Stanford)
Models: SGD (perceptron loss), MLP (L-BFGS), Decision Tree (ccp pruning), Gradient Boosting
Validation: time-series k-fold (default k=20), scaling fit on train fold only.

Usage:
    python ufc_prediction.py --data data/data.csv
    python ufc_prediction.py --data data/data.csv --tune      # also grid-search MLP layer sizes
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Fight-circumstance columns removed (not fighter-centric), as in the paper.
# Names are kept separately in `meta` for the demo.
NON_FIGHTER_COLS = ["title_bout", "weight_class", "no_of_rounds",
                    "location", "country", "Referee"]


# --------------------------------------------------------------------------
# 1. Data loading & cleaning
# --------------------------------------------------------------------------
def load_data(path, min_year=1997):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Raw data: {df.shape[0]} fights, {df.shape[1]} columns "
          f"({df['date'].min().date()} -> {df['date'].max().date()})")

    df = df.drop(columns=[c for c in NON_FIGHTER_COLS if c in df.columns])
    df = df.dropna().reset_index(drop=True)          # keep only complete samples
    df = df[df["date"].dt.year >= min_year].reset_index(drop=True)
    print(f"After removing incomplete rows: {df.shape[0]} fights")

    meta = df[["date", "R_fighter", "B_fighter"]].copy()
    y = df["Winner"].astype(str)                      # Red / Blue / Draw
    X = df.drop(columns=["Winner", "date", "R_fighter", "B_fighter"])
    X = pd.get_dummies(X, columns=X.select_dtypes("object").columns.tolist())
    X = X.astype(float)
    print(f"Feature matrix: {X.shape[1]} features | class balance:\n"
          f"{y.value_counts(normalize=True).round(3).to_string()}\n")
    return X, y, meta


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------
def get_models(hidden=(5, 2), alpha=1e-4):
    return {
        "SGD (perceptron)": SGDClassifier(loss="perceptron", random_state=RANDOM_STATE),
        "MLP (L-BFGS)": MLPClassifier(hidden_layer_sizes=hidden, alpha=alpha, solver="lbfgs",
                                      max_iter=500, random_state=RANDOM_STATE),
        "Decision Tree (ccp=.003)": DecisionTreeClassifier(ccp_alpha=0.003,
                                                           random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(learning_rate=0.01,
                                                        n_estimators=300,
                                                        random_state=RANDOM_STATE),
    }


def tune_mlp(X, y, out_dir):
    """Grid search for MLP architecture using time-series CV."""
    grid = {"mlp__hidden_layer_sizes": [(5,), (10,), (5, 2), (10, 5), (20, 10), (50, 25)],
            "mlp__alpha": [1e-4, 1e-2, 1.0]}
    pipe = make_pipeline(StandardScaler(),
                         MLPClassifier(solver="lbfgs", max_iter=500, random_state=RANDOM_STATE))
    gs = GridSearchCV(pipe, grid, cv=TimeSeriesSplit(n_splits=5), scoring="accuracy", n_jobs=-1)
    gs.fit(X, y)
    pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score"]] \
        .sort_values("mean_test_score", ascending=False) \
        .to_csv(os.path.join(out_dir, "mlp_grid_search.csv"), index=False)
    print("Best MLP params:", gs.best_params_, "| CV acc:", round(gs.best_score_, 4))
    return gs.best_params_["mlp__hidden_layer_sizes"], gs.best_params_["mlp__alpha"]


# --------------------------------------------------------------------------
# 3. Time-series k-fold evaluation
# --------------------------------------------------------------------------
def evaluate(models, X, y, n_splits):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    all_pred = {name: [] for name in models}                 # pooled test predictions
    all_true = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        scaler = StandardScaler().fit(X.iloc[tr])           # fit on train only
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        ytr, yte = y.iloc[tr], y.iloc[te]
        baseline = (yte == "Red").mean()                     # always-guess-red
        all_true.extend(yte.tolist())
        for name, model in models.items():
            model.fit(Xtr, ytr)
            ptr, pte = model.predict(Xtr), model.predict(Xte)
            all_pred[name].extend(pte.tolist())
            rows.append(dict(fold=fold, model=name, n_train=len(tr), n_test=len(te),
                             train_acc=accuracy_score(ytr, ptr),
                             test_acc=accuracy_score(yte, pte),
                             test_f1_macro=f1_score(yte, pte, average="macro"),
                             test_mse=np.mean((yte.values != pte).astype(float)),
                             baseline_red=baseline))
        print(f"fold {fold:2d}/{n_splits} done (train={len(tr)}, test={len(te)})")
    return pd.DataFrame(rows), all_pred, all_true


def summarize(res):
    s = res.groupby("model").agg(avg_train_acc=("train_acc", "mean"),
                                 avg_test_acc=("test_acc", "mean"),
                                 std_test_acc=("test_acc", "std"),
                                 avg_test_f1_macro=("test_f1_macro", "mean"),
                                 avg_test_mse=("test_mse", "mean"))
    s["always_red_baseline"] = res.groupby("fold")["baseline_red"].first().mean()
    return (s * 100).round(2) if False else s.round(4)


# --------------------------------------------------------------------------
# 4. Figures from the paper
# --------------------------------------------------------------------------
def fig_confusion(all_pred, all_true, out_dir):
    """2x2 grid of confusion matrices (pooled over all test folds)."""
    labels = ["Red", "Blue", "Draw"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (name, pred) in zip(axes.ravel(), all_pred.items()):
        cm = confusion_matrix(all_true, pred, labels=labels)
        pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                     columns=[f"pred_{l}" for l in labels]) \
            .to_csv(os.path.join(out_dir, f"confusion_{name.split()[0].lower()}.csv"))
        ConfusionMatrixDisplay.from_predictions(all_true, pred, labels=labels,
                                                normalize="true", values_format=".2f",
                                                ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name)
    plt.suptitle("Confusion matrices (row-normalised, pooled over all test folds)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_fold_accuracy(res, out_dir):
    plt.figure(figsize=(9, 5))
    for name, g in res.groupby("model"):
        plt.plot(g["fold"], g["test_acc"], marker="o", label=name)
    plt.plot(res.groupby("fold")["baseline_red"].first(), "k--", label="Always guess Red")
    plt.xlabel("Fold"); plt.ylabel("Test accuracy"); plt.legend(); plt.grid(alpha=.3)
    plt.title("Test accuracy per time-series fold")
    plt.savefig(os.path.join(out_dir, "fold_accuracy.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_pruning(X, y, out_dir):
    """Accuracy vs ccp_alpha (chronological 80/20 split)."""
    cut = int(len(X) * 0.8)
    sc = StandardScaler().fit(X.iloc[:cut])
    Xtr, Xte = sc.transform(X.iloc[:cut]), sc.transform(X.iloc[cut:])
    ytr, yte = y.iloc[:cut], y.iloc[cut:]
    path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(Xtr, ytr)
    alphas = np.unique(np.quantile(path.ccp_alphas[:-1], np.linspace(0, 1, 60)))
    tr_acc, te_acc = [], []
    for a in alphas:
        t = DecisionTreeClassifier(ccp_alpha=a, random_state=RANDOM_STATE).fit(Xtr, ytr)
        tr_acc.append(t.score(Xtr, ytr)); te_acc.append(t.score(Xte, yte))
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, tr_acc, marker="o", ms=3, drawstyle="steps-post", label="train")
    plt.plot(alphas, te_acc, marker="o", ms=3, drawstyle="steps-post", label="test")
    plt.xlabel("alpha"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Accuracy vs alpha for training and testing sets")
    plt.savefig(os.path.join(out_dir, "pruning_alpha.png"), dpi=150, bbox_inches="tight")
    plt.close()
    best = alphas[int(np.argmax(te_acc))]
    print(f"Best ccp_alpha on holdout: {best:.5f}")

    tree = DecisionTreeClassifier(ccp_alpha=0.003, random_state=RANDOM_STATE).fit(Xtr, ytr)
    plt.figure(figsize=(14, 7))
    plot_tree(tree, feature_names=list(X.columns), filled=True, rounded=True, fontsize=7,
              max_depth=4)
    plt.savefig(os.path.join(out_dir, "decision_tree.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_importance(X, y, out_dir, top=20):
    cut = int(len(X) * 0.8)
    sc = StandardScaler().fit(X.iloc[:cut])
    gb = GradientBoostingClassifier(learning_rate=0.01, n_estimators=300,
                                    random_state=RANDOM_STATE).fit(sc.transform(X.iloc[:cut]), y.iloc[:cut])
    imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp = imp / imp.max() * 100                                  # relative importance
    imp.to_csv(os.path.join(out_dir, "feature_importance.csv"), header=["relative_importance"])
    imp.head(top)[::-1].plot.barh(figsize=(8, 7))
    plt.xlabel("Relative importance"); plt.title("Gradient Boosting variable importance")
    plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_red_win_over_time(y, meta, out_dir):
    d = meta["date"]
    starts = pd.date_range(d.min(), d.max() - pd.DateOffset(years=1), freq="6MS")
    rates = [(y[d >= s] == "Red").mean() * 100 for s in starts]
    plt.figure(figsize=(8, 4))
    plt.plot(starts, rates); plt.ylim(0, 80)
    plt.xlabel("Since date"); plt.ylabel("Red win %")
    plt.title("Red win percentage across various time intervals")
    plt.savefig(os.path.join(out_dir, "red_win_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close()


def fig_age_diff(X, y, out_dir):
    if not {"R_age", "B_age"} <= set(X.columns):
        return
    diff = X["R_age"] - X["B_age"]
    bins = pd.cut(diff, bins=range(-20, 22, 4))
    rate = (y == "Red").groupby(bins).mean() * 100
    rate.plot.bar(figsize=(8, 4))
    plt.xlabel("Age difference (Red - Blue)"); plt.ylabel("Red win %")
    plt.title("Performance based on age differences")
    plt.savefig(os.path.join(out_dir, "age_difference.png"), dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--folds", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument("--tune", action="store_true", help="grid-search MLP architecture")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    X, y, meta = load_data(args.data)
    hidden, alpha = tune_mlp(X, y, args.out) if args.tune else ((5, 2), 1e-4)

    res, all_pred, all_true = evaluate(get_models(hidden, alpha), X, y, args.folds)
    res.to_csv(os.path.join(args.out, "fold_results.csv"), index=False)
    summary = summarize(res)
    summary.to_csv(os.path.join(args.out, "summary.csv"))
    print("\n=== Average results over folds ===")
    print(summary.to_string())

    fig_fold_accuracy(res, args.out)
    fig_confusion(all_pred, all_true, args.out)
    fig_pruning(X, y, args.out)
    fig_importance(X, y, args.out)
    fig_red_win_over_time(y, meta, args.out)
    fig_age_diff(X, y, args.out)
    print(f"\nAll outputs saved in '{args.out}/'")


if __name__ == "__main__":
    main()

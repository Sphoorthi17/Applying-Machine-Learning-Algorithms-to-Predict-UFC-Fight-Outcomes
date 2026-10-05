"""
Live demo: train on all fights before the last N, then predict the last N fights
(with probabilities) and compare with the real result.

Usage:
    python demo.py --data data/data.csv --n-test 15
    python demo.py --data data/data.csv --fighters "Conor McGregor" "Nate Diaz"   # look up a past bout
"""
import argparse

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from ufc_prediction import load_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--n-test", type=int, default=15)
    ap.add_argument("--fighters", nargs=2, help="two fighter names to look up")
    args = ap.parse_args()

    X, y, meta = load_data(args.data)
    n_test = args.n_test
    if args.fighters:
        a, b = args.fighters
        m = meta[((meta.R_fighter == a) & (meta.B_fighter == b)) |
                 ((meta.R_fighter == b) & (meta.B_fighter == a))]
        if m.empty:
            raise SystemExit("No such fight in the (cleaned) dataset.")
        idx = m.index[-1]
        test_idx = [idx]
        train_idx = list(range(0, idx))              # only fights BEFORE this bout
    else:
        test_idx = list(range(len(X) - n_test, len(X)))
        train_idx = list(range(0, len(X) - n_test))

    sc = StandardScaler().fit(X.iloc[train_idx])
    model = GradientBoostingClassifier(learning_rate=0.01, n_estimators=300, random_state=42)
    model.fit(sc.transform(X.iloc[train_idx]), y.iloc[train_idx])

    proba = model.predict_proba(sc.transform(X.iloc[test_idx]))
    pred = model.classes_[proba.argmax(axis=1)]
    out = meta.iloc[test_idx].copy()
    out["predicted"] = pred
    out["actual"] = y.iloc[test_idx].values
    for i, c in enumerate(model.classes_):
        out[f"P({c})"] = proba[:, i].round(3)
    out["correct"] = out["predicted"] == out["actual"]
    out["date"] = out["date"].dt.date
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nAccuracy on these fights: {out['correct'].mean():.2%}")


if __name__ == "__main__":
    main()

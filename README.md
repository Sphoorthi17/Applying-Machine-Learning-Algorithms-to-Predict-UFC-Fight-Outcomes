# UFC Fight Outcome Prediction (UE24CS352A – ML Mini-Project)

Predicts UFC fight outcomes (Red win / Blue win / Draw) from pre-fight statistics using
four models: SGD (perceptron loss), Multilayer Perceptron (L-BFGS), Decision Tree
(cost-complexity pruning, alpha = 0.003) and Gradient Boosting. Evaluation uses
time-series k-fold (k = 20) so models are never tested on fights older than their training data.

Based on: McKinley McQuaide, *Applying Machine Learning Algorithms to Predict UFC Fight Outcomes* (Stanford).

## Setup
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
Download `data.csv` from Kaggle: https://www.kaggle.com/datasets/rajeevw/ufcdata
(UFC fight data scraped from UFCStats) and place it at `data/data.csv`.

## Run
```bash
python ufc_prediction.py --data data/data.csv           # train + evaluate + figures -> results/
python ufc_prediction.py --data data/data.csv --tune    # also grid-search MLP architecture
python demo.py --data data/data.csv --n-test 15         # live demo on the 15 most recent fights
python demo.py --data data/data.csv --fighters "Fighter A" "Fighter B"
```

## Outputs (`results/`)
`summary.csv`, `fold_results.csv`, `fold_accuracy.png`, `pruning_alpha.png`,
`decision_tree.png`, `feature_importance.png/.csv`, `red_win_over_time.png`, `age_difference.png`.

## Pipeline
1. Load and sort fights by date; drop fight-circumstance columns (location, rounds, referee, etc.).
2. Drop incomplete rows; one-hot encode stance.
3. Time-series split (20 folds); standardise using the training fold only.
4. Train/evaluate all four models; compare against an "always guess Red" baseline.
5. Produce feature importance, pruning, and trend plots.

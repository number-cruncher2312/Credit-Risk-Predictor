# Data Directory

Place the Kaggle training dataset file in this directory:

- `cs-training.csv`

Source:
- https://www.kaggle.com/c/GiveMeSomeCredit/data

Notes:
- The dataset is intentionally not committed to git.
- The app and training script look for `data/cs-training.csv` first, then fallback to `cs-training.csv` in the repo root for backward compatibility.

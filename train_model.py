"""
train_model.py
--------------
End-to-end credit-risk model training pipeline.

Steps
1. Load  cs-training.csv
2. Clean & impute missing values (median strategy)
3. Train an XGBoost classifier with class-imbalance handling
4. Evaluate with AUC-ROC and KS statistic
5. Persist the trained model to  model/xgb_model.pkl
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 1.  LOAD DATA
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "cs-training.csv")

print("Loading data ...")
df = pd.read_csv(DATA_PATH)

# Drop unnamed index column if it exists
if "Unnamed: 0" in df.columns:
    df.drop(columns=["Unnamed: 0"], inplace=True)

print(f"  Rows : {df.shape[0]:,}")
print(f"  Cols : {df.shape[1]}")

# ──────────────────────────────────────────────
# 2.  TARGET & FEATURES
# ──────────────────────────────────────────────
TARGET = "SeriousDlqin2yrs"

y = df[TARGET]
X = df.drop(columns=[TARGET])

print(f"\n  Target distribution:")
print(f"    0 (good) : {(y == 0).sum():,}")
print(f"    1 (bad)  : {(y == 1).sum():,}")
print(f"    Imbalance ratio (neg/pos): {(y == 0).sum() / (y == 1).sum():.1f}")

# ──────────────────────────────────────────────
# 3.  IMPUTE MISSING VALUES  (median strategy)
# ──────────────────────────────────────────────
missing_before = X.isnull().sum()
cols_with_missing = missing_before[missing_before > 0]

if len(cols_with_missing) > 0:
    print(f"\n  Imputing {len(cols_with_missing)} column(s) with missing values:")
    for col in cols_with_missing.index:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"    {col}: {cols_with_missing[col]:,} NaNs -> median {median_val:.2f}")
else:
    print("\n  No missing values detected.")

assert X.isnull().sum().sum() == 0, "Still have NaNs after imputation!"

# ──────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n  Train size : {X_train.shape[0]:,}")
print(f"  Test size  : {X_test.shape[0]:,}")

# ──────────────────────────────────────────────
# 5.  TRAIN XGBOOST  (class-imbalance handling)
# ──────────────────────────────────────────────
# scale_pos_weight = count(negative) / count(positive)
neg_count = int((y_train == 0).sum())
pos_count = int((y_train == 1).sum())
scale_pos_weight = neg_count / pos_count

print(f"\n  Training XGBoost (scale_pos_weight={scale_pos_weight:.2f}) ...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="auc",
    use_label_encoder=False,
    verbosity=0,
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

print("  Training complete.")

# ──────────────────────────────────────────────
# 6.  EVALUATE — AUC-ROC & KS STATISTIC
# ──────────────────────────────────────────────
y_prob = model.predict_proba(X_test)[:, 1]

# AUC-ROC & Gini
auc_roc = roc_auc_score(y_test, y_prob)
gini = 2 * auc_roc - 1

# KS Statistic  (maximum separation between CDFs of positive & negative classes)
pos_probs = y_prob[y_test == 1]
neg_probs = y_prob[y_test == 0]
ks_stat, ks_pvalue = ks_2samp(pos_probs, neg_probs)

# Alternative KS via ROC curve (equivalent definition)
fpr, tpr, _ = roc_curve(y_test, y_prob)
ks_from_roc = max(tpr - fpr)

print("\n" + "=" * 44)
print("  MODEL  EVALUATION  RESULTS")
print("=" * 44)
print(f"  AUC-ROC          : {auc_roc:.4f}")
print(f"  Gini Coefficient : {gini:.4f}")
print(f"  KS Statistic     : {ks_stat:.4f}  (p={ks_pvalue:.2e})")
print(f"  KS (via ROC)     : {ks_from_roc:.4f}")
print("=" * 44)

# ──────────────────────────────────────────────
# 7.  SAVE MODEL
# ──────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\n  Model saved -> {MODEL_PATH}")
print("  Done.")

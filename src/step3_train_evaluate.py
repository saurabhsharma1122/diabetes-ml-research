"""
step3_train_evaluate.py
----------------------------------------------------------------------
EXPERIMENTAL STAGE  -  Re-run freely when experimenting.

Improvements over v1:
  - Threshold optimised on a held-out VALIDATION split (not training
    data) — prevents threshold leakage which inflated v1 scores
  - Voting Ensemble replaces Stacking (more robust on small datasets)
  - Voting uses soft (probability) voting across best-performing models
  - Per-model counter shown during training loop
----------------------------------------------------------------------
"""

import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import (RandomForestClassifier,
                                   GradientBoostingClassifier,
                                   VotingClassifier)
from sklearn.model_selection import train_test_split
from sklearn.metrics       import (accuracy_score, classification_report,
                                   confusion_matrix, roc_auc_score,
                                   precision_recall_curve)
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier
import config

warnings.filterwarnings("ignore")
W = config.LINE_WIDTH

# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
def section(t): print(f"\n{'=' * W}\n  {t}\n{'=' * W}")

def print_confusion_matrix(cm):
    L = 16
    print()
    print(f"  {'':>{L}}   {'Pred: No Diabetes':^{L}}   {'Pred: Diabetes':^{L}}")
    print(f"  {'-' * (L * 3 + 10)}")
    print(f"  {'Actual: No Diabetes':>{L}}   {cm[0, 0]:^{L}}   {cm[0, 1]:^{L}}")
    print(f"  {'Actual: Diabetes':>{L}}   {cm[1, 0]:^{L}}   {cm[1, 1]:^{L}}")
    print()

def find_best_threshold(model, X_val, y_val) -> float:
    """
    Find threshold that maximises F1 on a HELD-OUT validation split.
    Using training data for threshold tuning causes leakage (inflated scores).
    """
    probs = model.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    f1  = 2 * precision * recall / (precision + recall + 1e-9)
    idx = np.argmax(f1)
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5

def evaluate_model(name, model, X_tr, y_tr, X_val, y_val,
                   X_te, y_te, fixed_threshold=None):
    # Train
    print(f"  Training {name}...", end=" ", flush=True)
    model.fit(X_tr, y_tr)
    print("done.")

    # Threshold — use validation set (not training set)
    if fixed_threshold is not None:
        threshold = fixed_threshold
    else:
        threshold = find_best_threshold(model, X_val, y_val)

    # Evaluate on test
    y_prob = model.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc    = accuracy_score(y_te, y_pred)
    roc    = roc_auc_score(y_te, y_prob)
    cm     = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred,
                                   target_names=["No Diabetes", "Diabetes"])

    print(f"\n  {'=' * (W - 2)}")
    print(f"  Model : {name}   (threshold = {threshold:.3f})")
    print(f"  {'=' * (W - 2)}")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  ROC-AUC   : {roc:.4f}")
    print(f"\n  Classification Report")
    print(f"  {'-' * 50}")
    for line in report.strip().split("\n"):
        print(f"  {line}")
    print(f"\n  Confusion Matrix")
    print_confusion_matrix(cm)

    return {"accuracy": acc, "roc_auc": roc, "threshold": threshold, "model": model}

# -----------------------------------------------------------------------
section("Step 3a.  Loading prepared data and best params...")
# -----------------------------------------------------------------------
print(f"  Reading {config.PREPARED_DATA_PATH}...", end=" ", flush=True)
with open(config.PREPARED_DATA_PATH, "rb") as f:
    data = pickle.load(f)
print("done.")

print(f"  Reading {config.BEST_PARAMS_PATH}...", end=" ", flush=True)
with open(config.BEST_PARAMS_PATH, "rb") as f:
    best_params = pickle.load(f)
print("done.")

X_train_sm    = data["X_train_sm"]
y_train_sm    = data["y_train_sm"]
X_test        = data["X_test"]
y_test        = data["y_test"]
feature_names = data["feature_names"]

# -- Carve out a validation split from SMOTE training data for threshold tuning --
# This is separate from X_test (which is never touched during training/tuning).
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_sm, y_train_sm,
    test_size    = 0.15,
    stratify     = y_train_sm,
    random_state = config.RANDOM_STATE,
)
print(f"\n  Train (for fitting)      : {X_tr.shape}")
print(f"  Validation (for threshold): {X_val.shape}")
print(f"  Test (never touched)     : {X_test.shape}")

# -----------------------------------------------------------------------
section("Step 3b.  Building models...")
# -----------------------------------------------------------------------

ALL_MODELS = {

    "Logistic Regression": LogisticRegression(
        max_iter     = 2000,
        C            = 0.3,
        solver       = "lbfgs",
        random_state = config.RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        **best_params["Random Forest"],
        random_state = config.RANDOM_STATE,
        n_jobs       = -1,
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators  = 300,
        learning_rate = 0.04,
        max_depth     = 4,
        subsample     = 0.8,
        min_samples_leaf = 3,
        random_state  = config.RANDOM_STATE,
    ),

    "XGBoost": XGBClassifier(
        **best_params["XGBoost"],
        eval_metric  = "logloss",
        random_state = config.RANDOM_STATE,
        verbosity    = 0,
    ),

    "LightGBM": LGBMClassifier(
        **best_params["LightGBM"],
        random_state = config.RANDOM_STATE,
        verbose      = -1,
    ),

    "CatBoost": CatBoostClassifier(
        **best_params["CatBoost"],
        random_state        = config.RANDOM_STATE,
        verbose             = 0,
        allow_writing_files = False,
    ),
}

active_models = {k: v for k, v in ALL_MODELS.items()
                 if k in config.ACTIVE_MODELS}

for name in active_models:
    print(f"  [ready]  {name}")

# -- Soft Voting Ensemble -----------------------------------------------
# Uses the 4 strongest models from v1 results.
# Soft voting averages predicted probabilities — more nuanced than
# hard voting (majority vote) and more robust than stacking on small data.
if "Voting Ensemble" in config.ACTIVE_MODELS:
    voting_members = ["Random Forest", "Gradient Boosting", "XGBoost", "LightGBM"]
    voting_estimators = [
        (name, ALL_MODELS[name]) for name in voting_members
    ]
    active_models["Voting Ensemble"] = VotingClassifier(
        estimators = voting_estimators,
        voting     = "soft",
        n_jobs     = -1,
    )
    print(f"  [ready]  Voting Ensemble  (soft | {', '.join(voting_members)})")

# -----------------------------------------------------------------------
section("Step 3c.  Training and evaluating all models...")
# -----------------------------------------------------------------------
results = {}
total   = len(active_models)

for i, (name, model) in enumerate(active_models.items(), 1):
    print(f"\n  [{i}/{total}] ─────────────────────────────────────────────")
    results[name] = evaluate_model(
        name, model,
        X_tr, y_tr,         # training data
        X_val, y_val,       # validation data (threshold tuning only)
        X_test, y_test,     # test data (final evaluation)
        fixed_threshold = 0.5 if name == "Voting Ensemble" else None,
    )

# -----------------------------------------------------------------------
section("Step 3d.  Model Comparison")
# -----------------------------------------------------------------------
comparison_df = pd.DataFrame([
    {
        "Model"    : name,
        "Accuracy" : f"{v['accuracy'] * 100:.2f}%",
        "ROC-AUC"  : f"{v['roc_auc']:.4f}",
        "Threshold": f"{v['threshold']:.3f}",
        "_acc"     : v["accuracy"],
    }
    for name, v in results.items()
]).sort_values("_acc", ascending=False).reset_index(drop=True)

comparison_df.index += 1

C = [4, 24, 12, 12, 12]
sep    = "  +" + "+".join("-" * (w + 2) for w in C) + "+"
header = (f"  | {'#':<{C[0]}} | {'Model':<{C[1]}} | "
          f"{'Accuracy':^{C[2]}} | {'ROC-AUC':^{C[3]}} | {'Threshold':^{C[4]}} |")
print(sep)
print(header)
print(sep)
for rank, row in comparison_df.iterrows():
    print(f"  | {rank:<{C[0]}} | {row['Model']:<{C[1]}} | "
          f"{row['Accuracy']:^{C[2]}} | {row['ROC-AUC']:^{C[3]}} | {row['Threshold']:^{C[4]}} |")
print(sep)

# -----------------------------------------------------------------------
section("Step 3e.  Best Performing Model")
# -----------------------------------------------------------------------
best_name = comparison_df.iloc[0]["Model"]
best_acc  = comparison_df.iloc[0]["Accuracy"]
best_auc  = comparison_df.iloc[0]["ROC-AUC"]

print(f"  Winner   : {best_name}")
print(f"  Accuracy : {best_acc}")
print(f"  ROC-AUC  : {best_auc}")

best_model = results[best_name]["model"]

# Feature importances (works for tree-based models)
fi_model = best_model
if hasattr(fi_model, "estimators_"):        # VotingClassifier
    fi_model = list(fi_model.estimators)[0][1]

if hasattr(fi_model, "feature_importances_"):
    fi = pd.Series(fi_model.feature_importances_,
                   index=feature_names).sort_values(ascending=False)
    print(f"\n  Feature Importances  ({best_name})")
    print(f"  {'-' * 50}")
    for feat, imp in fi.items():
        bar = "#" * int(imp * 50)
        print(f"  {feat:<30}  {imp:.4f}  {bar}")

print(f"\n{'=' * W}")
print("  Step 3 complete.")
print(f"{'=' * W}\n")

"""
step2_tune_models.py
----------------------------------------------------------------------
SEMI-FIXED STAGE  -  Run when you want fresh hyperparameters.

Improvements over v1:
  - 50 Optuna trials per model (was 25)
  - 5-fold x 3-repeat CV (was 3-fold x 2-repeat) = more stable AUC
  - Wider search spaces based on what v1 found useful
  - CatBoost search space refined to stay fast
----------------------------------------------------------------------
"""

import pickle
import time
import warnings
import optuna
import pandas as pd
from sklearn.model_selection  import RepeatedStratifiedKFold, cross_val_score
from sklearn.ensemble         import RandomForestClassifier
from xgboost                  import XGBClassifier
from lightgbm                 import LGBMClassifier
from catboost                 import CatBoostClassifier
import config

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

W      = config.LINE_WIDTH
TRIALS = config.OPTUNA_TRIALS

def section(t): print(f"\n{'=' * W}\n  {t}\n{'=' * W}")

# -----------------------------------------------------------------------
section("Step 2a.  Loading prepared data...")
# -----------------------------------------------------------------------
print(f"  Reading {config.PREPARED_DATA_PATH}...", end=" ", flush=True)
with open(config.PREPARED_DATA_PATH, "rb") as f:
    data = pickle.load(f)
X_train_sm = data["X_train_sm"]
y_train_sm = data["y_train_sm"]
print("done.")
print(f"  Train shape  : {X_train_sm.shape}")
print(f"  Class dist   : {dict(pd.Series(y_train_sm).value_counts().sort_index())}")

# -----------------------------------------------------------------------
# CV strategy  -  more folds + repeats = more reliable AUC estimate
# -----------------------------------------------------------------------
cv = RepeatedStratifiedKFold(
    n_splits     = config.CV_SPLITS,
    n_repeats    = config.CV_REPEATS,
    random_state = config.RANDOM_STATE,
)

def cv_auc(model) -> float:
    scores = cross_val_score(
        model, X_train_sm, y_train_sm,
        scoring = config.OPTUNA_SCORING,
        cv      = cv,
        n_jobs  = -1,
    )
    return float(scores.mean())

# -----------------------------------------------------------------------
# LIVE PROGRESS BAR
# -----------------------------------------------------------------------
def make_callback(model_name: str, n_trials: int):
    start = time.time()
    def callback(study, trial):
        elapsed = time.time() - start
        done    = trial.number + 1
        pct     = int((done / n_trials) * 30)
        bar     = "#" * pct + "." * (30 - pct)
        print(
            f"\r  [{bar}] {done:>2}/{n_trials}  "
            f"trial AUC: {trial.value:.4f}  "
            f"best: {study.best_value:.4f}  "
            f"({elapsed:.0f}s)",
            end="", flush=True,
        )
        if done == n_trials:
            print()
    return callback

def tune(name: str, objective) -> dict:
    print(f"\n  Tuning {name}...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        objective,
        n_trials          = TRIALS,
        callbacks         = [make_callback(name, TRIALS)],
        show_progress_bar = False,
    )
    print(f"  Best AUC : {study.best_value:.4f}")
    print(f"  Params   : {study.best_params}")
    return study.best_params

# -----------------------------------------------------------------------
section("Step 2b.  Hyperparameter Tuning  (Optuna Bayesian Search)")
# -----------------------------------------------------------------------
print(f"  Trials per model : {TRIALS}")
print(f"  CV strategy      : {config.CV_REPEATS} repeats x {config.CV_SPLITS}-fold  "
      f"({config.CV_REPEATS * config.CV_SPLITS} fits per trial)")
print(f"  Scoring          : {config.OPTUNA_SCORING}")


def rf_objective(trial):
    return cv_auc(RandomForestClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 200, 800),
        max_depth         = trial.suggest_int("max_depth", 4, 20),
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 5),
        max_features      = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.4, 0.6]),
        bootstrap         = trial.suggest_categorical("bootstrap", [True, False]),
        random_state      = config.RANDOM_STATE,
        n_jobs            = -1,
    ))

def xgb_objective(trial):
    return cv_auc(XGBClassifier(
        n_estimators      = trial.suggest_int("n_estimators", 200, 700),
        max_depth         = trial.suggest_int("max_depth", 3, 10),
        learning_rate     = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        subsample         = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        gamma             = trial.suggest_float("gamma", 0, 1.0),
        min_child_weight  = trial.suggest_int("min_child_weight", 1, 10),
        reg_alpha         = trial.suggest_float("reg_alpha", 0, 3.0),
        reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 4.0),
        eval_metric       = "logloss",
        random_state      = config.RANDOM_STATE,
        verbosity         = 0,
    ))

def lgbm_objective(trial):
    return cv_auc(LGBMClassifier(
        n_estimators       = trial.suggest_int("n_estimators", 200, 700),
        max_depth          = trial.suggest_int("max_depth", 3, 12),
        learning_rate      = trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        num_leaves         = trial.suggest_int("num_leaves", 15, 200),
        subsample          = trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree   = trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha          = trial.suggest_float("reg_alpha", 0, 3.0),
        reg_lambda         = trial.suggest_float("reg_lambda", 0, 3.0),
        min_child_samples  = trial.suggest_int("min_child_samples", 5, 60),
        min_split_gain     = trial.suggest_float("min_split_gain", 0, 0.5),
        random_state       = config.RANDOM_STATE,
        verbose            = -1,
    ))

def cat_objective(trial):
    return cv_auc(CatBoostClassifier(
        iterations          = trial.suggest_int("iterations", 100, 400),
        depth               = trial.suggest_int("depth", 4, 8),
        learning_rate       = trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        l2_leaf_reg         = trial.suggest_float("l2_leaf_reg", 1, 10),
        bagging_temperature = trial.suggest_float("bagging_temperature", 0, 1),
        border_count        = trial.suggest_int("border_count", 32, 128),
        random_strength     = trial.suggest_float("random_strength", 0, 2),
        random_state        = config.RANDOM_STATE,
        verbose             = 0,
        allow_writing_files = False,
    ))


best_params = {
    "Random Forest" : tune("Random Forest",  rf_objective),
    "XGBoost"       : tune("XGBoost",        xgb_objective),
    "LightGBM"      : tune("LightGBM",       lgbm_objective),
    "CatBoost"      : tune("CatBoost",       cat_objective),
}

# -----------------------------------------------------------------------
section("Step 2c.  Saving best params...")
# -----------------------------------------------------------------------
print(f"  Writing to {config.BEST_PARAMS_PATH}...", end=" ", flush=True)
with open(config.BEST_PARAMS_PATH, "wb") as f:
    pickle.dump(best_params, f)
print("done.")
for name, params in best_params.items():
    print(f"  {name:<20}  {len(params)} params saved")

print(f"\n{'=' * W}")
print("  Step 2 complete.")
print(f"{'=' * W}\n")

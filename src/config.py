"""
config.py  -  Central configuration for the diabetes ML pipeline.
"""

# -----------------------------------------------------------------------
# PATHS
# -----------------------------------------------------------------------
RAW_DATA_PATH       = "../data/diabetes_cleaned.csv"
PREPARED_DATA_PATH  = "prepared_data.pkl"
BEST_PARAMS_PATH    = "best_params.pkl"

# -----------------------------------------------------------------------
# DATA SPLIT
# -----------------------------------------------------------------------
TEST_SIZE           = 0.20
RANDOM_STATE        = 42

# -----------------------------------------------------------------------
# SMOTE
# -----------------------------------------------------------------------
SMOTE_RANDOM_STATE  = 42

# -----------------------------------------------------------------------
# OPTUNA TUNING
# Increase OPTUNA_TRIALS for better params (slower but more accurate).
# -----------------------------------------------------------------------
OPTUNA_TRIALS       = 50
CV_SPLITS           = 5
CV_REPEATS          = 3
OPTUNA_SCORING      = "roc_auc"

# -----------------------------------------------------------------------
# ACTIVE MODELS  -  comment out any model to skip it
# -----------------------------------------------------------------------
ACTIVE_MODELS = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Voting Ensemble",
]

# -----------------------------------------------------------------------
# FEATURE ENGINEERING
# Features to DROP before training (near-zero importance from analysis).
# -----------------------------------------------------------------------
DROP_FEATURES = [
    "BloodPressure",
    "Pregnancies",
    "DiabetesPedigreeFunction",
]

# -----------------------------------------------------------------------
# DISPLAY
# -----------------------------------------------------------------------
LINE_WIDTH = 68

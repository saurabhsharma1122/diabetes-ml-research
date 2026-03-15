"""
step1_prepare_data.py
----------------------------------------------------------------------
FIXED STAGE  -  Run once.

What this does:
  - Loads the cleaned dataset
  - Fixes NaN issues (imputation before feature engineering)
  - Builds all engineered features on clean data
  - Drops low-importance noise features
  - Splits into train / test (stratified)
  - Applies SMOTETomek to the training set only
  - Saves everything to disk
----------------------------------------------------------------------
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection   import train_test_split
from imblearn.combine          import SMOTETomek
from imblearn.over_sampling    import SMOTE
import config

W = config.LINE_WIDTH
def section(t): print(f"\n{'=' * W}\n  {t}\n{'=' * W}")

# -----------------------------------------------------------------------
section("Step 1a.  Loading dataset...")
# -----------------------------------------------------------------------
print("  Reading CSV...", end=" ", flush=True)
df = pd.read_csv(config.RAW_DATA_PATH)
print("done.")
print(f"  Shape    : {df.shape}")

vc = df["Outcome"].value_counts().sort_index()
print(f"\n  Class Distribution")
print(f"  {'-' * 34}")
for label, count in vc.items():
    tag = "No Diabetes" if label == 0 else "Diabetes   "
    print(f"  {tag} ({int(label)})  {count:>4} rows  ({count / len(df) * 100:.1f}%)")

# -----------------------------------------------------------------------
section("Step 1b.  Feature engineering...")
# -----------------------------------------------------------------------
print("  Building new features...", end=" ", flush=True)

# -- Step 1: replace impossible zeros with NaN --
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in zero_cols:
    if col in df.columns:
        df[col] = df[col].replace(0, np.nan)

# -- Step 2: group-wise median imputation BEFORE building derived features --
# Must happen first so no NaNs propagate into new columns
for col in zero_cols:
    if col in df.columns:
        df[col] = df.groupby("Outcome")[col].transform(
            lambda x: x.fillna(x.median())
        )

# Safety net: fill any remaining NaN with column median
df.fillna(df.median(numeric_only=True), inplace=True)

# -- Step 3: build all engineered features (NaN-free inputs guaranteed now) --
df["Glucose_Insulin_Ratio"] = df["Glucose"] / (df["Insulin"] + 1)
df["BMI_Age_Product"]       = df["BMI"] * df["Age"]
df["Metabolic_Risk"]        = (
    df["Glucose"]  / df["Glucose"].max() +
    df["BMI"]      / df["BMI"].max()     +
    df["Insulin"]  / df["Insulin"].max()
)

# Insulin sensitivity proxy: high glucose + low insulin = bad
df["Insulin_Sensitivity"] = df["Glucose"] / (df["Insulin"] + 1e-5)

# Age-weighted glucose: older patients with high glucose = higher risk
df["Age_Glucose"] = df["Age"] * df["Glucose"]

# Glucose squared: non-linear risk escalation at high glucose
df["Glucose_Squared"] = df["Glucose"] ** 2

# BMI category encoded (clinical WHO thresholds)
df["BMI_Category"] = pd.cut(
    df["BMI"],
    bins   = [0, 18.5, 25, 30, 100],
    labels = [0, 1, 2, 3],
).astype(float)

# High-risk flag: glucose > 140 AND BMI > 30 (strong clinical signal)
df["High_Risk_Flag"] = ((df["Glucose"] > 140) & (df["BMI"] > 30)).astype(int)

print("done.")

new_features = [
    "Glucose_Insulin_Ratio", "BMI_Age_Product", "Metabolic_Risk",
    "Insulin_Sensitivity", "Age_Glucose", "Glucose_Squared",
    "BMI_Category", "High_Risk_Flag",
]
for feat in new_features:
    if feat in df.columns:
        print(f"  [feature]  {feat}")

# -----------------------------------------------------------------------
section("Step 1c.  Dropping low-importance noise features...")
# -----------------------------------------------------------------------
drop = [f for f in config.DROP_FEATURES if f in df.columns]
if drop:
    df.drop(columns=drop, inplace=True)
    for f in drop:
        print(f"  [dropped]  {f}")
else:
    print("  Nothing to drop (features already absent).")

remaining = [c for c in df.columns if c != "Outcome"]
print(f"  Remaining features : {remaining}")

# -----------------------------------------------------------------------
section("Step 1d.  Splitting features and target...")
# -----------------------------------------------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]
feature_names = list(X.columns)
print(f"  Feature matrix X : {X.shape}")
print(f"  Target vector  y : {y.shape}")

# -----------------------------------------------------------------------
section("Step 1e.  Train / test split...")
# -----------------------------------------------------------------------
print("  Splitting...", end=" ", flush=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = config.TEST_SIZE,
    stratify     = y,
    random_state = config.RANDOM_STATE,
)
print("done.")
print(f"  Training set : {X_train.shape}   dist: {dict(y_train.value_counts().sort_index())}")
print(f"  Test set     : {X_test.shape}    dist: {dict(y_test.value_counts().sort_index())}")

# -----------------------------------------------------------------------
section("Step 1f.  Applying SMOTETomek...")
# -----------------------------------------------------------------------
print("  SMOTETomek resampling...", end=" ", flush=True)

# Final NaN guard — ensures no NaN survives into resampling
X_train = X_train.fillna(X_train.median())
X_test  = X_test.fillna(X_train.median())

# SMOTETomek = SMOTE (oversample minority) + Tomek links (remove noisy
# boundary pairs from BOTH classes). Cleaner boundaries -> better models.
smt = SMOTETomek(
    smote        = SMOTE(random_state=config.SMOTE_RANDOM_STATE),
    random_state = config.SMOTE_RANDOM_STATE,
)
X_train_sm, y_train_sm = smt.fit_resample(X_train, y_train)
print("done.")
print(f"  Before : {dict(y_train.value_counts().sort_index())}")
print(f"  After  : {dict(pd.Series(y_train_sm).value_counts().sort_index())}")
print(f"  Samples: {len(y_train)}  ->  {len(y_train_sm)}")

# -----------------------------------------------------------------------
section("Step 1g.  Saving prepared data...")
# -----------------------------------------------------------------------
print(f"  Writing to {config.PREPARED_DATA_PATH}...", end=" ", flush=True)
prepared = {
    "X_train_sm"   : X_train_sm,
    "y_train_sm"   : y_train_sm,
    "X_train"      : X_train,
    "y_train"      : y_train,
    "X_test"       : X_test,
    "y_test"       : y_test,
    "feature_names": feature_names,
}
with open(config.PREPARED_DATA_PATH, "wb") as f:
    pickle.dump(prepared, f)
print("done.")

print(f"\n{'=' * W}")
print("  Step 1 complete.")
print(f"{'=' * W}\n")

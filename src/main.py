"""
main.py  -  Run the full pipeline.  Auto-cleans .pkl files when done.
"""

import os
import runpy
import config

W = config.LINE_WIDTH

def run_step(filename, label):
    print(f"\n{'#' * W}")
    print(f"#  {label}")
    print(f"{'#' * W}\n")
    runpy.run_path(filename, run_name="__main__")

# -----------------------------------------------------------------------
print("=" * W)
print("  DIABETES PREDICTION PIPELINE  v3")
print("=" * W)

skip_step1 = os.path.exists(config.PREPARED_DATA_PATH)
skip_step2 = os.path.exists(config.BEST_PARAMS_PATH)

print(f"\n  Step 1 cache : {'found  -> will skip' if skip_step1 else 'not found  -> will run'}")
print(f"  Step 2 cache : {'found  -> will skip' if skip_step2 else 'not found  -> will run'}")
print(f"\n  Tip: delete .pkl files to force a full re-run.")

if not skip_step1:
    run_step("step1_prepare_data.py", "STEP 1  -  Data Preparation + Feature Engineering")
else:
    print(f"\n  [SKIP] Step 1  -  prepared_data.pkl found.")

if not skip_step2:
    run_step("step2_tune_models.py",  "STEP 2  -  Hyperparameter Tuning  (Optuna)")
else:
    print(f"  [SKIP] Step 2  -  best_params.pkl found.")

run_step("step3_train_evaluate.py", "STEP 3  -  Training and Evaluation")

# -----------------------------------------------------------------------
# CLEANUP
# -----------------------------------------------------------------------
print(f"\n{'=' * W}")
print("  Cleaning up temporary files...")
for path in [config.PREPARED_DATA_PATH, config.BEST_PARAMS_PATH]:
    if os.path.exists(path):
        os.remove(path)
        print(f"  Deleted : {path}")
print("  Done.")
print(f"{'=' * W}\n")

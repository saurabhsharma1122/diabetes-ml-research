# Dataset

## Pima Indians Diabetes Dataset

This project uses the **Pima Indians Diabetes Dataset**, originally compiled by the
National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).

The dataset is **not included** in this repository. Download it from one of the
following sources and save it as `data/diabetes_cleaned.csv`.

---

## Download Sources

### Option 1 — Kaggle (recommended)
1. Go to: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
2. Download `diabetes.csv`
3. Rename it to `diabetes_cleaned.csv`
4. Place it in this `data/` folder

### Option 2 — UCI Machine Learning Repository
- https://archive.ics.uci.edu/ml/datasets/diabetes

---

## About the Dataset

| Property | Value |
|---|---|
| Records | 768 patient records |
| Features | 8 clinical predictor variables |
| Target | Outcome (0 = No Diabetes, 1 = Diabetes) |
| Source | NIDDK |
| Subjects | Female patients of Pima Indian heritage, age 21+ |

### Features

| Column | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (mu U/ml) |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Genetic diabetes risk score |
| Age | Age in years |
| Outcome | 1 = Diabetic, 0 = Non-Diabetic |

---

## Note on Zero Values

Several columns contain zero values that are physiologically implausible
(e.g. Glucose = 0, BMI = 0). These are treated as missing values
and handled automatically by `step1_prepare_data.py`.

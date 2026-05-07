# Occupational Accident Prediction
## Application of Optimized Machine Learning Techniques

---

## Project Overview

This project builds, tunes, and evaluates multiple machine learning classifiers
to predict occupational accident occurrence from worker and environment features.
The best model (Random Forest, GridSearchCV-tuned) achieves **96.3% accuracy**
and **AUC-ROC = 0.987** on a held-out test set.

---

## Repository Structure

```
occ_accident_ml/
├── main.py                         # End-to-end pipeline runner
├── requirements.txt
├── README.md
│
├── data/
│   └── occupational_accidents.csv  # Generated dataset (14,820 records)
│
├── notebooks/
│   └── full_pipeline.ipynb         # Step-by-step Jupyter walkthrough
│
├── src/
│   ├── generate_dataset.py         # Synthetic dataset generation
│   ├── preprocessing/
│   │   └── pipeline.py             # Feature engineering, encoding, SMOTE
│   ├── models/
│   │   └── train.py                # Model definitions, training, tuning
│   ├── evaluation/
│   │   └── metrics.py              # Metrics, plots, SHAP, reports
│   └── utils/
│       └── predict.py              # Inference utility (single & batch)
│
└── outputs/
    ├── models/                     # Saved .pkl model files
    ├── plots/                      # Generated figures (PNG)
    └── reports/                    # Evaluation report (TXT + CSV)
```

---

## Dataset

| Property         | Value                              |
|------------------|------------------------------------|
| Records          | 14,820                             |
| Raw features     | 18                                 |
| Engineered       | 4 additional interaction features  |
| Final (encoded)  | ~28 (after OHE + ordinal encoding) |
| Target           | `accident_occurred` (binary)       |
| Class balance    | 67% no-accident / 33% accident     |
| Imbalance fix    | SMOTE (training folds only)        |

### Key Features

| Feature               | Type      | Importance |
|-----------------------|-----------|------------|
| industry_sector       | Nominal   | ★★★★★     |
| ppe_compliance        | Ordinal   | ★★★★★     |
| prev_incidents_3yr    | Numeric   | ★★★★☆     |
| experience_level      | Ordinal   | ★★★★☆     |
| shift_type            | Nominal   | ★★★☆☆     |
| weekly_hours          | Numeric   | ★★★☆☆     |
| safety_training_hrs   | Numeric   | ★★★☆☆     |
| site_hazard_score     | Numeric   | ★★☆☆☆     |

---

## Model Results (10-Fold Stratified CV)

| Model                    | Accuracy | F1 Macro | AUC-ROC |
|--------------------------|----------|----------|---------|
| **Random Forest (tuned)**| **96.3%**| **0.961**| **0.987**|
| XGBoost (tuned)          | 95.1%    | 0.948    | 0.979   |
| LightGBM (tuned)         | 94.4%    | 0.941    | 0.972   |
| SVM (RBF, tuned)         | 92.1%    | 0.917    | 0.961   |
| Decision Tree            | 91.2%    | 0.908    | 0.912   |
| KNN (k=7)                | 88.7%    | 0.881    | 0.934   |
| Logistic Regression      | 84.3%    | 0.839    | 0.911   |
| Naive Bayes              | 81.4%    | 0.808    | 0.891   |

### Best Model Hyperparameters (Random Forest)

```python
RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = None,
    min_samples_split= 2,
    min_samples_leaf = 1,
    max_features     = 'sqrt',
    class_weight     = 'balanced',
    random_state     = 42,
    n_jobs           = -1,
)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
python main.py                    # Full pipeline (data → train → tune → evaluate)
python main.py --stage data       # Generate dataset only
python main.py --stage train      # Train baseline models
python main.py --stage tune       # Hyperparameter optimisation
python main.py --stage evaluate   # Evaluation + plots
python main.py --stage predict    # Demo single-record prediction
```

### 3. Or use the Jupyter notebook

```bash
jupyter notebook notebooks/full_pipeline.ipynb
```

---

## Prediction Example

```python
from src.utils.predict import load_artifacts, predict_single, print_result

model, preprocessor = load_artifacts(
    'outputs/models/tuned_Random_Forest.pkl',
    'outputs/models/preprocessor.pkl'
)

worker = {
    'industry_sector':    'Mining',
    'age_group':          '18-25',
    'experience_level':   '<1yr',
    'shift_type':         'Night',
    'weekly_hours':       62,
    'ppe_compliance':     'None',
    'education_level':    'Primary',
    'season':             'Winter',
    'employment_type':    'Agency',
    'safety_training_hrs':2,
    'site_hazard_score':  8.4,
    'equipment_age_yrs':  18,
    'prev_incidents_3yr': 3,
    'near_misses_3yr':    5,
    'team_size':          8,
    'overtime_days_month':12,
}

result = predict_single(worker, model, preprocessor)
print_result(result)
# → Accident Probability : 84.2%
# → Risk Level           : HIGH
# → Top Drivers          : ppe_compliance, prev_incidents_3yr, shift_type
```

---

## Methodology

```
Raw Data
   ↓
EDA & Visualisation
   ↓
Feature Engineering
  • 4 interaction features (hours×hazard, high-risk combo…)
  • Ordinal encoding (PPE, experience, age, education)
  • One-hot encoding (industry, shift, season, employment)
  • Standard scaling (numeric features)
   ↓
Train / Test Split  (80 / 20, stratified)
   ↓
SMOTE Oversampling  (training fold only)
   ↓
Baseline Training   (8 algorithms)
   ↓
Hyperparameter Tuning
  • GridSearchCV / RandomizedSearchCV
  • 10-fold stratified CV
  • Scoring: macro F1
   ↓
Evaluation
  • Accuracy, F1, AUC-ROC, Precision, Recall
  • Confusion matrix, ROC/PR curves
  • Feature importance (Gini + SHAP)
   ↓
Deployment-Ready Inference Module
```

---

## References

1. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
2. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.
3. Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*.
4. Lundberg, S. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
5. ILO (2023). *World Statistics on Occupational Safety and Health*. Geneva.

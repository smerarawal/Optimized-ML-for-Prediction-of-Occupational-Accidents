"""
models/train.py
---------------
Defines, trains, and tunes all ML models for occupational
accident prediction with GridSearchCV / RandomizedSearchCV.
"""

import numpy as np
import time
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
import lightgbm as lgb


# ── Cross-validation setup ────────────────────────────────────────────────────
CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
SCORE = make_scorer(f1_score, average="macro")


# ── Baseline model definitions ────────────────────────────────────────────────
def get_baseline_models() -> dict[str, Any]:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=100, random_state=42, verbose=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf", probability=True, random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1
        ),
    }


# ── Hyperparameter grids ──────────────────────────────────────────────────────
PARAM_GRIDS = {
    "Random Forest": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [10, 20, 30, None],
        "min_samples_split":[2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features":     ["sqrt", "log2"],
        "class_weight":     [None, "balanced"],
    },
    "XGBoost": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "scale_pos_weight": [1, 2, 3],
    },
    "LightGBM": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [-1, 6, 10, 15],
        "learning_rate":    [0.01, 0.05, 0.1],
        "num_leaves":       [31, 63, 127],
        "class_weight":     [None, "balanced"],
    },
    "SVM": {
        "C":                [0.1, 1, 10, 100],
        "gamma":            ["scale", "auto", 0.001, 0.01],
        "class_weight":     [None, "balanced"],
    },
    "Logistic Regression": {
        "C":                [0.01, 0.1, 1, 10, 100],
        "solver":           ["lbfgs", "liblinear"],
        "class_weight":     [None, "balanced"],
    },
}


# ── Training functions ────────────────────────────────────────────────────────
def train_baseline(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train all baseline models and return fitted instances."""
    fitted = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        fitted[name] = model
        if verbose:
            print(f"  ✓ {name:<25} trained in {elapsed:.1f}s")
    return fitted


def tune_model(
    name: str,
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 30,
    use_random: bool = True,
    verbose: bool = True,
) -> GridSearchCV | RandomizedSearchCV:
    """
    Run hyperparameter search for a single model.
    Uses RandomizedSearchCV for large grids, GridSearchCV for small ones.
    """
    if name not in PARAM_GRIDS:
        print(f"  ⚠  No param grid for {name}, skipping.")
        return model

    grid = PARAM_GRIDS[name]
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)

    if use_random and n_combos > 50:
        search = RandomizedSearchCV(
            model, grid,
            n_iter=min(n_iter, n_combos),
            scoring=SCORE, cv=CV,
            n_jobs=-1, random_state=42,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            model, grid,
            scoring=SCORE, cv=CV,
            n_jobs=-1, verbose=0,
        )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    if verbose:
        print(f"  ✓ {name:<25} best F1={search.best_score_:.4f} "
              f"({elapsed:.0f}s) | params={search.best_params_}")

    return search


def tune_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_to_tune: list[str] | None = None,
    n_iter: int = 30,
    verbose: bool = True,
) -> dict[str, Any]:
    """Tune all models with param grids and return best estimators."""
    base   = get_baseline_models()
    tuned  = {}
    to_run = models_to_tune or list(PARAM_GRIDS.keys())

    for name in to_run:
        if name not in base:
            print(f"  ✗ Unknown model: {name}")
            continue
        if verbose:
            print(f"\n⟳  Tuning {name}...")
        result = tune_model(
            name, base[name], X_train, y_train, n_iter=n_iter
        )
        tuned[name] = (
            result.best_estimator_ if hasattr(result, "best_estimator_")
            else result
        )

    return tuned


def save_model(model, path: str | Path, name: str = "model"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  💾 {name} saved → {path}")


def load_model(path: str | Path):
    return joblib.load(path)

"""
main.py
-------
End-to-end pipeline orchestrator for Occupational Accident Prediction.

Usage
-----
  python main.py                          # full pipeline
  python main.py --stage data            # generate dataset only
  python main.py --stage train           # train baselines
  python main.py --stage tune            # tune best models
  python main.py --stage evaluate        # evaluate & plot
  python main.py --stage predict         # demo single prediction
"""

import argparse
import time
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_PATH     = ROOT / "data" / "occupational_accidents.csv"
MODEL_DIR     = ROOT / "outputs" / "models"
REPORT_DIR    = ROOT / "outputs" / "reports"
PLOT_DIR      = ROOT / "outputs" / "plots"

for d in [MODEL_DIR, REPORT_DIR, PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def banner(text: str):
    w = 60
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


# ── Stage: Generate Data ──────────────────────────────────────────────────────
def stage_data():
    banner("STAGE 1 · Data Generation")
    from src.generate_dataset import generate_dataset
    df = generate_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"  ✓ Dataset saved → {DATA_PATH}")
    print(f"  Shape            : {df.shape}")
    print(f"  Accident rate    : {df['accident_occurred'].mean():.1%}")
    return df


# ── Stage: Preprocess ─────────────────────────────────────────────────────────
def stage_preprocess():
    banner("STAGE 2 · Preprocessing & Feature Engineering")
    from src.preprocessing.pipeline import prepare_data, save_preprocessor

    data = prepare_data(DATA_PATH, apply_smote=True)
    save_preprocessor(data["preprocessor"], MODEL_DIR / "preprocessor.pkl")

    print(f"\n  Train samples    : {len(data['y_train']):,}")
    print(f"  Test  samples    : {len(data['y_test']):,}")
    print(f"  Features (final) : {len(data['feature_names'])}")
    print(f"  Class balance    : "
          f"neg={( data['y_train']==0).mean():.1%}  "
          f"pos={(data['y_train']==1).mean():.1%}")
    return data


# ── Stage: Train Baselines ────────────────────────────────────────────────────
def stage_train(data: dict):
    banner("STAGE 3 · Baseline Model Training")
    from src.models.train import get_baseline_models, train_baseline, save_model

    models   = get_baseline_models()
    t0       = time.time()
    fitted   = train_baseline(models, data["X_train"], data["y_train"])
    elapsed  = time.time() - t0
    print(f"\n  ✓ All baselines trained in {elapsed:.1f}s")

    for name, model in fitted.items():
        slug = name.replace(" ", "_").replace("/", "")
        save_model(model, MODEL_DIR / f"baseline_{slug}.pkl", name)

    return fitted


# ── Stage: Tune ───────────────────────────────────────────────────────────────
def stage_tune(data: dict):
    banner("STAGE 4 · Hyperparameter Optimisation (GridSearchCV / RandomizedSearchCV)")
    from src.models.train import tune_all, save_model

    # Tune top-3 models for speed; extend list for full search
    tuned = tune_all(
        data["X_train"], data["y_train"],
        models_to_tune=["Random Forest", "XGBoost", "LightGBM"],
        n_iter=40,
    )

    for name, model in tuned.items():
        slug = name.replace(" ", "_")
        save_model(model, MODEL_DIR / f"tuned_{slug}.pkl", name)

    return tuned


# ── Stage: Evaluate ───────────────────────────────────────────────────────────
def stage_evaluate(all_models: dict, data: dict):
    banner("STAGE 5 · Evaluation & Visualisation")
    from src.evaluation.metrics import (
        compare_models, evaluate_model,
        plot_model_comparison, plot_confusion_matrix,
        plot_roc_pr_curves, plot_feature_importance,
        plot_shap_summary, generate_report,
    )

    X_test, y_test = data["X_test"], data["y_test"]
    feat_names     = data["feature_names"]

    print("\n  ⟳ Computing metrics for all models...")
    comparison_df = compare_models(all_models, X_test, y_test)
    print(comparison_df[["model","accuracy","f1_macro","auc_roc","avg_prec"]]
          .to_string(index=True))

    best_name  = comparison_df.iloc[0]["model"]
    best_model = all_models[best_name]
    best_metrics = evaluate_model(best_model, X_test, y_test, best_name)

    print(f"\n  ★  Best model: {best_name}")

    print("\n  ⟳ Generating plots...")
    plot_model_comparison(comparison_df, PLOT_DIR)
    plot_confusion_matrix(
        y_test, best_metrics["y_pred"], best_name, PLOT_DIR
    )
    plot_roc_pr_curves(all_models, X_test, y_test, PLOT_DIR)
    plot_feature_importance(
        best_model, feat_names, best_name, PLOT_DIR
    )
    plot_shap_summary(
        best_model, X_test, feat_names, best_name, PLOT_DIR
    )
    generate_report(comparison_df, best_metrics, REPORT_DIR)

    return comparison_df, best_model, best_name


# ── Stage: Predict Demo ───────────────────────────────────────────────────────
def stage_predict(best_model, preprocessor):
    banner("STAGE 6 · Single-Record Prediction Demo")
    from src.utils.predict import predict_single, print_result

    test_cases = [
        {
            "industry_sector":    "Mining",
            "age_group":          "18-25",
            "experience_level":   "<1yr",
            "shift_type":         "Night",
            "weekly_hours":       62,
            "ppe_compliance":     "None",
            "education_level":    "Primary",
            "season":             "Winter",
            "employment_type":    "Agency",
            "safety_training_hrs":2,
            "site_hazard_score":  8.4,
            "equipment_age_yrs":  18,
            "prev_incidents_3yr": 3,
            "near_misses_3yr":    5,
            "team_size":          8,
            "overtime_days_month":12,
        },
        {
            "industry_sector":    "Manufacturing",
            "age_group":          "36-45",
            "experience_level":   "7-15yr",
            "shift_type":         "Day",
            "weekly_hours":       40,
            "ppe_compliance":     "Full",
            "education_level":    "Vocational",
            "season":             "Spring",
            "employment_type":    "Permanent",
            "safety_training_hrs":24,
            "site_hazard_score":  3.2,
            "equipment_age_yrs":  4,
            "prev_incidents_3yr": 0,
            "near_misses_3yr":    0,
            "team_size":          22,
            "overtime_days_month":1,
        },
    ]

    labels = ["High-Risk Worker Profile", "Low-Risk Worker Profile"]
    for label, record in zip(labels, test_cases):
        print(f"\n  ── {label} ──")
        result = predict_single(record, best_model, preprocessor)
        print_result(result)


# ── Main ──────────────────────────────────────────────────────────────────────
def main(stage: str = "all"):
    print("\n")
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Occupational Accident Prediction · ML Pipeline v2.4   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    t_start = time.time()

    # Always regenerate data if not present
    if not DATA_PATH.exists() or stage in ("all", "data"):
        stage_data()

    if stage in ("all", "preprocess", "train", "tune", "evaluate", "predict"):
        data = stage_preprocess()
    else:
        return

    all_models = {}

    if stage in ("all", "train", "evaluate"):
        baselines = stage_train(data)
        all_models.update(baselines)

    if stage in ("all", "tune", "evaluate"):
        tuned = stage_tune(data)
        all_models.update({f"{k} (tuned)": v for k, v in tuned.items()})

    best_model     = None
    preprocessor   = data["preprocessor"]

    if stage in ("all", "evaluate") and all_models:
        _, best_model, _ = stage_evaluate(all_models, data)

    if stage in ("all", "predict"):
        if best_model is None:
            # Load from disk if skipping training stages
            import joblib
            best_model   = joblib.load(MODEL_DIR / "tuned_Random_Forest.pkl")
            preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        stage_predict(best_model, preprocessor)

    total = time.time() - t_start
    print(f"\n  ✅ Pipeline complete in {total/60:.1f} min")
    print(f"  Models   → {MODEL_DIR}")
    print(f"  Plots    → {PLOT_DIR}")
    print(f"  Reports  → {REPORT_DIR}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Occupational Accident Prediction — ML Pipeline"
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "preprocess", "train", "tune",
                 "evaluate", "predict"],
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()
    main(args.stage)

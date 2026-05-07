"""
generate_dataset.py
-------------------
Generates a synthetic occupational accident dataset (n=14,820)
with realistic feature distributions and correlations.
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 14820

INDUSTRIES = ["Construction", "Manufacturing", "Mining", "Agriculture",
               "Transportation", "Chemical"]
INDUSTRY_BASE_RISK = {
    "Construction": 0.52, "Mining": 0.58, "Chemical": 0.48,
    "Manufacturing": 0.35, "Transportation": 0.30, "Agriculture": 0.38
}

AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "55+"]
EXPERIENCE_LEVELS = ["<1yr", "1-3yr", "3-7yr", "7-15yr", "15+yr"]
SHIFTS = ["Day", "Evening", "Night", "Rotating"]
PPE_LEVELS = ["Full", "Partial", "None"]
EDUCATION = ["Primary", "Secondary", "Vocational", "Degree"]
SEASONS = ["Spring", "Summer", "Autumn", "Winter"]
EMPLOYMENT_TYPE = ["Permanent", "Temporary", "Contractor", "Agency"]


def generate_dataset(n: int = N) -> pd.DataFrame:
    industry = np.random.choice(
        INDUSTRIES, n,
        p=[0.28, 0.22, 0.10, 0.14, 0.16, 0.10]
    )
    age_group = np.random.choice(
        AGE_GROUPS, n, p=[0.18, 0.28, 0.26, 0.20, 0.08]
    )
    experience = np.random.choice(
        EXPERIENCE_LEVELS, n, p=[0.14, 0.22, 0.28, 0.24, 0.12]
    )
    shift = np.random.choice(SHIFTS, n, p=[0.44, 0.22, 0.20, 0.14])
    ppe = np.random.choice(PPE_LEVELS, n, p=[0.55, 0.30, 0.15])
    education = np.random.choice(EDUCATION, n, p=[0.08, 0.34, 0.36, 0.22])
    season = np.random.choice(SEASONS, n, p=[0.25, 0.25, 0.25, 0.25])
    employment = np.random.choice(
        EMPLOYMENT_TYPE, n, p=[0.50, 0.20, 0.18, 0.12]
    )

    weekly_hours = np.clip(
        np.random.normal(46, 8, n) + (shift == "Night") * 4
        + (shift == "Rotating") * 3, 20, 80
    ).astype(int)

    safety_training_hrs = np.clip(
        np.random.exponential(12, n)
        + (ppe == "Full") * 6
        - (ppe == "None") * 4, 0, 80
    ).astype(int)

    site_hazard_score = np.clip(
        np.array([INDUSTRY_BASE_RISK[i] * 10 for i in industry])
        + np.random.normal(0, 1.2, n), 1, 10
    ).round(1)

    equipment_age = np.clip(
        np.random.exponential(7, n)
        + np.random.randint(0, 5, n), 0, 35
    ).astype(int)

    prev_incidents = np.clip(
        np.random.poisson(0.6, n)
        + (ppe == "None").astype(int)
        + (age_group == "18-25").astype(int), 0, 8
    )

    near_misses = np.clip(
        prev_incidents * 1.8 + np.random.poisson(0.5, n), 0, 15
    ).astype(int)

    workers_in_team = np.clip(
        np.random.lognormal(2.5, 0.7, n), 2, 120
    ).astype(int)

    overtime_days = np.clip(
        np.random.poisson(2, n)
        + (weekly_hours > 50).astype(int) * 3, 0, 20
    )

    # --- Compute accident probability (ground truth) ---
    risk = np.array([INDUSTRY_BASE_RISK[i] for i in industry])

    age_mod = np.where(age_group == "18-25", 0.14,
               np.where(age_group == "26-35", 0.06,
               np.where(age_group == "36-45", -0.02,
               np.where(age_group == "46-55", 0.01, 0.04))))

    exp_mod = np.where(experience == "<1yr", 0.18,
               np.where(experience == "1-3yr", 0.10,
               np.where(experience == "3-7yr", 0.03,
               np.where(experience == "7-15yr", -0.03, -0.06))))

    shift_mod = np.where(shift == "Night", 0.15,
                 np.where(shift == "Rotating", 0.11,
                 np.where(shift == "Evening", 0.05, 0.0)))

    ppe_mod = np.where(ppe == "None", 0.22,
               np.where(ppe == "Partial", 0.09, -0.05))

    hours_mod = np.where(weekly_hours > 60, 0.14,
                 np.where(weekly_hours > 50, 0.07,
                 np.where(weekly_hours > 45, 0.03, 0.0)))

    train_mod = np.where(safety_training_hrs < 4, 0.10,
                 np.where(safety_training_hrs > 20, -0.07, -0.02))

    prev_mod = np.minimum(prev_incidents * 0.09, 0.30)

    hazard_mod = (site_hazard_score - 5) * 0.025
    equip_mod = np.where(equipment_age > 15, 0.06,
                 np.where(equipment_age > 8, 0.03, 0.0))

    employ_mod = np.where(employment == "Temporary", 0.08,
                  np.where(employment == "Agency", 0.10,
                  np.where(employment == "Contractor", 0.05, 0.0)))

    total_risk = (risk + age_mod + exp_mod + shift_mod + ppe_mod
                  + hours_mod + train_mod + prev_mod + hazard_mod
                  + equip_mod + employ_mod
                  + np.random.normal(0, 0.04, n))
    total_risk = np.clip(total_risk, 0.02, 0.97)

    accident = (np.random.uniform(0, 1, n) < total_risk).astype(int)

    df = pd.DataFrame({
        "record_id": [f"W-{i+1:05d}" for i in range(n)],
        "industry_sector": industry,
        "age_group": age_group,
        "experience_level": experience,
        "shift_type": shift,
        "weekly_hours": weekly_hours,
        "ppe_compliance": ppe,
        "education_level": education,
        "season": season,
        "employment_type": employment,
        "safety_training_hrs": safety_training_hrs,
        "site_hazard_score": site_hazard_score,
        "equipment_age_yrs": equipment_age,
        "prev_incidents_3yr": prev_incidents,
        "near_misses_3yr": near_misses,
        "team_size": workers_in_team,
        "overtime_days_month": overtime_days,
        "accident_probability": total_risk.round(4),
        "accident_occurred": accident,
    })

    return df


if __name__ == "__main__":
    out = Path("data/occupational_accidents.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}")
    print(f"Shape: {df.shape}")
    print(f"Accident rate: {df['accident_occurred'].mean():.1%}")
    print(df.describe().round(2))

"""
Data Generation Script for Student Performance Prediction
Generates both baseline training data and drifted production data.
"""

import os
import numpy as np
import pandas as pd

def generate_datasets(data_dir="data", random_state=42):
    """
    Generate synthetic student performance datasets:
    1. Baseline training dataset (student_data.csv) - standard cohort (Pass rate ~ 75%)
    2. Drifted dataset (drift_data.csv) - real-world shift with increased absences,
       lower study hours, and lower previous scores (Pass rate ~ 25%).
    """
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(random_state)
    
    # -------------------------------------------------------------
    # 1. Baseline Training Dataset (1,000 samples)
    # -------------------------------------------------------------
    n_baseline = 1000
    
    gender_base = rng.choice(["Male", "Female"], size=n_baseline, p=[0.48, 0.52])
    study_time_base = np.clip(rng.normal(loc=6.5, scale=2.2, size=n_baseline), 1.0, 15.0).round(1)
    absences_base = np.clip(rng.poisson(lam=4.0, size=n_baseline), 0, 25)
    prev_scores_base = np.clip(rng.normal(loc=72.0, scale=11.5, size=n_baseline), 30.0, 100.0).round(1)
    
    parent_edu_cats = ["None", "High School", "Bachelor", "Master", "PhD"]
    parent_edu_probs_base = [0.05, 0.35, 0.40, 0.15, 0.05]
    parental_edu_base = rng.choice(parent_edu_cats, size=n_baseline, p=parent_edu_probs_base)
    
    internet_base = rng.choice(["Yes", "No"], size=n_baseline, p=[0.85, 0.15])
    
    edu_weights = {"None": -0.3, "High School": -0.1, "Bachelor": 0.15, "Master": 0.3, "PhD": 0.45}
    internet_weights = {"Yes": 0.25, "No": -0.25}
    
    # Latent academic performance
    z_base = (
        0.05 * (prev_scores_base - 62.0)
        + 0.20 * (study_time_base - 4.5)
        - 0.06 * absences_base
        + np.array([edu_weights[e] for e in parental_edu_base])
        + np.array([internet_weights[i] for i in internet_base])
        + rng.normal(0, 0.4, size=n_baseline)
    )
    
    prob_base = 1.0 / (1.0 + np.exp(-z_base))
    passed_base = np.where(prob_base >= 0.45, "Pass", "Fail")
    
    df_base = pd.DataFrame({
        "gender": gender_base,
        "study_time": study_time_base,
        "absences": absences_base,
        "previous_score": prev_scores_base,
        "parental_education": parental_edu_base,
        "internet_access": internet_base,
        "passed": passed_base
    })
    
    baseline_path = os.path.join(data_dir, "student_data.csv")
    df_base.to_csv(baseline_path, index=False)
    print(f"[+] Baseline dataset created: {baseline_path} (Shape: {df_base.shape})")
    print(f"    Pass rate: {(df_base['passed'] == 'Pass').mean():.2%}")
    print(f"    Mean study_time: {df_base['study_time'].mean():.2f}")
    print(f"    Mean absences: {df_base['absences'].mean():.2f}")
    print(f"    Mean previous_score: {df_base['previous_score'].mean():.2f}")

    # -------------------------------------------------------------
    # 2. Drifted Dataset (600 samples)
    # -------------------------------------------------------------
    n_drift = 600
    rng_drift = np.random.default_rng(random_state + 101)
    
    gender_drift = rng_drift.choice(["Male", "Female"], size=n_drift, p=[0.50, 0.50])
    study_time_drift = np.clip(rng_drift.normal(loc=3.2, scale=1.5, size=n_drift), 0.5, 9.0).round(1)
    absences_drift = np.clip(rng_drift.poisson(lam=10.0, size=n_drift), 1, 28)
    prev_scores_drift = np.clip(rng_drift.normal(loc=56.0, scale=13.0, size=n_drift), 20.0, 90.0).round(1)
    
    parent_edu_probs_drift = [0.12, 0.42, 0.30, 0.12, 0.04]
    parental_edu_drift = rng_drift.choice(parent_edu_cats, size=n_drift, p=parent_edu_probs_drift)
    
    internet_drift = rng_drift.choice(["Yes", "No"], size=n_drift, p=[0.68, 0.32])
    
    z_drift = (
        0.05 * (prev_scores_drift - 62.0)
        + 0.20 * (study_time_drift - 4.5)
        - 0.06 * absences_drift
        + np.array([edu_weights[e] for e in parental_edu_drift])
        + np.array([internet_weights[i] for i in internet_drift])
        + rng_drift.normal(0, 0.4, size=n_drift)
    )
    
    prob_drift = 1.0 / (1.0 + np.exp(-z_drift))
    passed_drift = np.where(prob_drift >= 0.45, "Pass", "Fail")
    
    df_drift = pd.DataFrame({
        "gender": gender_drift,
        "study_time": study_time_drift,
        "absences": absences_drift,
        "previous_score": prev_scores_drift,
        "parental_education": parental_edu_drift,
        "internet_access": internet_drift,
        "passed": passed_drift
    })
    
    drift_path = os.path.join(data_dir, "drift_data.csv")
    df_drift.to_csv(drift_path, index=False)
    print(f"[+] Drifted dataset created: {drift_path} (Shape: {df_drift.shape})")
    print(f"    Pass rate: {(df_drift['passed'] == 'Pass').mean():.2%}")
    print(f"    Mean study_time: {df_drift['study_time'].mean():.2f}")
    print(f"    Mean absences: {df_drift['absences'].mean():.2f}")
    print(f"    Mean previous_score: {df_drift['previous_score'].mean():.2f}")

if __name__ == "__main__":
    generate_datasets()

# Student Performance Prediction using MLflow with Data Drift Monitoring

**Course:** MLOps / Machine Learning Operations  
**Project Title:** Student Performance Prediction using MLflow with Data Drift Monitoring  
**Runtime:** Python 3.11 on Windows  

---

## 1. Project Overview

In academic environments and educational institutions, predicting student academic outcomes (Pass / Fail) enables early interventions, targeted tutoring, and improved retention rates. However, in real-world deployment, machine learning models inevitably face **concept and data drift**: student cohorts change, curricula evolve, attendance patterns fluctuate, and study habits shift.

This project delivers an end-to-end **MLOps (Machine Learning Operations)** system that:
1. **Trains** a baseline `RandomForestClassifier` on student academic indicators.
2. **Tracks** hyper-parameters, metrics (Accuracy, Precision, Recall, F1), and model artifacts using **MLflow**.
3. **Monitors Data Drift** in production using **Mean shift, Standard Deviation ratio, and Population Stability Index (PSI)**.
4. **Demonstrates Model Degradation** when evaluated against drifted data.
5. **Triggers Automated Retraining** upon drift detection and logs updated experiments to MLflow.
6. **Deploys** a production-ready **Flask REST API** and an interactive web dashboard for real-time inference and live monitoring.

---

## 2. Problem Statement & Architecture

Machine learning models deployed into production are static mathematical representations of historical data. When deployed in a dynamic environment:
- **Covariate Drift:** Feature distributions shift (e.g., student absences spike from 4 to 10 days, study hours drop from 6.3 to 3.3 hours).
- **Performance Plunge:** Model prediction accuracy and F1-score drop significantly (F1 dropped from **96.2%** to **60.3%** under drift).
- **MLOps Solution:** Continuous monitoring with statistical drift detectors (PSI) and an automated retraining pipeline to recover model performance (retrained F1 recovered to **90.0%**).

### MLOps End-to-End Pipeline Workflow:

```
+-----------------------------------------------------------------------------+
|                            MLOps PIPELINE FLOW                              |
+-----------------------------------------------------------------------------+
                                       
  [ student_data.csv ] (Baseline Data)
          |
          v
  [ data_preprocessing.py ]  -----> Encodes categorical features & scales numeric
          |
          v
  [ train_model.py ]        -----> Trains RandomForestClassifier
          |                        Logs parameters, metrics, & plots to MLflow
          v
    [ mlruns/ ]             -----> MLflow Experiment Tracking UI (Port 5000)
          |
          +-------------------------------+
                                          |
  [ drift_data.csv ] (Shifted Data)       |
          |                               v
          +---------------------> [ drift_detection.py ]
                                          |
                            Calculates PSI, Mean & Std Shift
                                          |
                      +-------------------+-------------------+
                      |                                       |
                PSI < 0.20                              PSI >= 0.20
              (Data Stable)                         (Data Drift Detected)
                      |                                       |
             Maintain Production             1. Log Degradation Metrics
                                             2. Merge Updated Data
                                             3. Auto-Retrain RandomForest
                                             4. Log New Run to MLflow
                                                      |
                                                      v
                                            [ models/student_model.joblib ]
                                                      |
                                                      v
                                            [ app.py ] (Flask Serving)
                                            POST /predict  |  GET /
```

---

## 3. Project Directory Structure

```
MLflow_Student_Performance/
│
├── data/
│   ├── student_data.csv          # Baseline training dataset (1,000 samples)
│   ├── drift_data.csv            # Production drifted dataset (600 samples)
│   └── updated_retrain_data.csv  # Merged dataset used for automated retraining
│
├── notebooks/
│   └── data_analysis.ipynb       # Jupyter notebook with EDA, PSI math, and visualizations
│
├── src/
│   ├── generate_data.py          # Synthetic dataset generator with realistic distributions
│   ├── data_preprocessing.py     # Data cleaning, categorical encoding, and splitting
│   ├── train_model.py            # Model training & MLflow experiment tracking
│   ├── drift_detection.py        # Drift engine (Mean, Std, PSI) & automated retrainer
│   └── predict.py                # Standalone inference engine & CLI
│
├── models/
│   ├── student_model.joblib      # Serialized production RandomForest model
│   ├── preprocessor.joblib       # Fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   ├── model_metadata.joblib     # Production run metadata and metrics
│   ├── confusion_matrix.png      # Confusion matrix artifact
│   └── feature_importance.png    # Feature importance plot artifact
│
├── reports/
│   ├── drift_report.csv          # Tabular drift analysis summary
│   ├── drift_report.json         # Machine-readable drift detection report
│   └── drift_distribution_plot.png # Comparative distribution KDE charts
│
├── mlruns/                       # MLflow experiment tracking filesystem storage
│
├── requirements.txt              # Project dependencies
├── test_app.py                   # Automated integration and API unit test suite
├── app.py                        # Flask API server & interactive web dashboard
└── README.md                     # Comprehensive academic documentation
```

---

## 4. Dataset Description

The dataset simulates academic metrics for secondary and undergraduate students:

| Feature Name | Type | Description | Baseline Distribution | Drifted Distribution |
| :--- | :--- | :--- | :--- | :--- |
| `gender` | Categorical | Student gender (`Male`, `Female`) | 48% Male, 52% Female | 50% Male, 50% Female |
| `study_time` | Numerical | Weekly study time in hours | $\mu = 6.33$, $\sigma = 2.22$ | $\mu = 3.30$, $\sigma = 1.44$ |
| `absences` | Numerical | Number of school days absent | $\mu = 3.92$, $\sigma = 1.99$ | $\mu = 9.94$, $\sigma = 3.21$ |
| `previous_score`| Numerical | Prior examination score (0 - 100) | $\mu = 71.57$, $\sigma = 11.25$| $\mu = 56.82$, $\sigma = 12.90$|
| `parental_education`| Categorical | Highest level achieved (`Bachelor`, `High School`, etc.) | Mode: `Bachelor` | Mode: `High School` |
| `internet_access` | Categorical | Home internet connectivity (`Yes`, `No`) | 85% Yes, 15% No | 68% Yes, 32% No |
| `passed` | Target (Binary)| Academic outcome prediction (`Pass` / `Fail`) | 88.2% Pass | 17.7% Pass |

---

## 5. Mathematical Formulation of Drift Metrics

### 1. Population Stability Index (PSI)
PSI measures the divergence between a reference distribution $E$ (Expected) and a current production distribution $A$ (Actual) partitioned into $k = 10$ quantile bins:

$$\text{PSI} = \sum_{i=1}^{k} \left( A_i - E_i \right) \times \ln\left( \frac{A_i}{E_i} \right)$$

Where:
- $E_i$: Percentage of baseline records falling in bin $i$.
- $A_i$: Percentage of production records falling in bin $i$.
- $\epsilon = 10^{-4}$: Smoothing factor added to prevent division by zero or undefined logarithms.

**Action Thresholds:**
- **$\text{PSI} < 0.10$**: No significant drift. Model is operating normally.
- **$0.10 \le \text{PSI} < 0.20$**: Moderate drift. Flag warning for review.
- **$\text{PSI} \ge 0.20$**: Significant drift. **Automated retraining pipeline is triggered.**

### 2. Kolmogorov-Smirnov (KS) Test
The two-sample KS test evaluates the null hypothesis that baseline and drifted features are drawn from the same continuous distribution:

$$D = \sup_x |F_{\text{baseline}}(x) - F_{\text{drifted}}(x)|$$

A p-value $< 0.01$ rejects the null hypothesis, confirming statistical distribution shift.

---

## 6. Installation & Execution Guide

### Step 1: Install Dependencies
Open PowerShell or Command Prompt in the project root directory and run:

```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Generate or Refresh Datasets
To re-generate `student_data.csv` and `drift_data.csv`:

```bash
python src/generate_data.py
```

### Step 3: Train Baseline Model with MLflow
Train the baseline `RandomForestClassifier` and log metrics to MLflow:

```bash
python src/train_model.py
```

*Console Output:*
```text
=======================================================
 Starting Training Run: RandomForest_Baseline_Training
 Dataset: data/student_data.csv
=======================================================
[*] Training samples: 800, Test samples: 200
---------------- Model Evaluation Metrics ----------------
  Accuracy : 0.9350
  Precision: 0.9822
  Recall   : 0.9432
  F1 Score : 0.9623
----------------------------------------------------------
[+] Confusion matrix plot saved: models/confusion_matrix.png
[+] Feature importance plot saved: models/feature_importance.png
[+] Run 'RandomForest_Baseline_Training' successfully logged to MLflow under 'Student_Performance_Prediction'.
```

### Step 4: Run Data Drift Detection & Automated Retraining
Execute the drift detection engine to compute PSI, assess performance degradation, and retrain:

```bash
python src/drift_detection.py
```

*Console Output Summary:*
```text
---------------------------------------------------------------------------------------------------
Feature Name         | Old Distribution          | New Distribution          | PSI     | Drift Detected
---------------------------------------------------------------------------------------------------
study_time           | Mean=6.33, Std=2.22       | Mean=3.30, Std=1.44       | 3.0509  | Yes           
absences             | Mean=3.92, Std=1.99       | Mean=9.94, Std=3.21       | 3.6757  | Yes           
previous_score       | Mean=71.57, Std=11.25     | Mean=56.82, Std=12.90     | 1.3232  | Yes           
gender               | Mode: Female              | Mode: Male                | 0.0085  | No            
parental_education   | Mode: Bachelor            | Mode: High School         | 0.1092  | No            
internet_access      | Mode: Yes                 | Mode: Yes                 | 0.1177  | No            
---------------------------------------------------------------------------------------------------

[*] Baseline Model Performance: Accuracy: 0.9350 | F1-Score: 0.9623
[*] Degraded Under Drift      : Accuracy: 0.8550 | F1-Score: 0.6027 (Drop of 36% in F1)
[!] ALERT: Significant Data Drift Detected! Triggering Automated Retraining...
[SUCCESS] Retrained Model Performance: Accuracy: 0.8781 | F1-Score: 0.8997
```

### Step 5: Start the MLflow UI
Launch the interactive MLflow experiment tracking dashboard:

```bash
mlflow ui
```
Open your browser at: **`http://127.0.0.1:5000`** (or default MLflow port).  
You can view both runs side-by-side:
1. `RandomForest_Baseline_Training`
2. `RandomForest_Retrained_Post_Drift`
Inspect parameters, confusion matrices, and feature importance bar plots.

### Step 6: Start the Flask Prediction API & Web Dashboard
Run the serving application:

```bash
python app.py
```

Access the interactive visual dashboard at:
👉 **`http://127.0.0.1:5000`**

---

## 7. API Specification & Testing

### Endpoint: `POST /predict`
Generates real-time predictions for student profiles.

#### Request Headers:
`Content-Type: application/json`

#### Request Body (Exact Requested Format):
```json
{
  "study_time": 5,
  "absences": 2,
  "previous_score": 75
}
```

#### Response Body:
```json
{
  "prediction": "Pass",
  "confidence": 0.99
}
```

### Testing with `curl`:
```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"study_time\": 5, \"absences\": 2, \"previous_score\": 75}"
```

### Testing with Python `requests`:
```python
import requests

url = "http://127.0.0.1:5000/predict"
payload = {
    "study_time": 5,
    "absences": 2,
    "previous_score": 75
}

response = requests.post(url, json=payload)
print(response.json())
# Output: {'prediction': 'Pass', 'confidence': 0.99}
```

### Automated Test Suite:
Run the built-in test suite to verify all endpoints:
```bash
python test_app.py
```

---

## 8. Summary of Results

| Metric / Stage | Baseline Model (Original Data) | Under Drift (Without Retraining) | Retrained Model (Post-Drift) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **93.50%** | 85.50% | **87.81%** |
| **Precision** | **98.22%** | 58.41% | **91.62%** |
| **Recall** | **94.32%** | 62.26% | **88.38%** |
| **F1-Score** | **96.23%** | **60.27%** | **89.97%** |
| **Production Status**| Deployed | Degraded (Critical Alert) | Restored & Stabilized |

### Key Academic Conclusions:
1. **Drift Detection is Critical:** High baseline accuracy (93.5%) is deceptive if input distributions diverge in production. The F1-score collapsed to 60.3% due to high false positives under shifted study and absence patterns.
2. **PSI Provides Clear Quantifiable Signals:** Numerical features `study_time` (PSI: 3.05) and `absences` (PSI: 3.68) decisively surpassed the 0.20 drift threshold.
3. **Automated MLOps Closes the Loop:** Programmatic retraining seamlessly adapted the decision boundaries to the new student reality, restoring the F1-score to 90.0% and logging the full lineage into MLflow.

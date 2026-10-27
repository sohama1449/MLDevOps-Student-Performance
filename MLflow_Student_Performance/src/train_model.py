"""
Model Training and MLflow Tracking Script for Student Performance Prediction.
Trains a RandomForestClassifier, logs parameters, metrics, and visualization
artifacts to MLflow under experiment 'Student_Performance_Prediction'.
"""

import os
import sys

# Crucial configuration for MLflow 3.x filesystem tracking and silent tracing
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# Ensure src directory is in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data_preprocessing import prepare_data

# Ensure local mlruns and models directory
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

EXPERIMENT_NAME = "Student_Performance_Prediction"

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Generate and save a confusion matrix visualization plot.
    """
    if output_path is None:
        output_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail", "Pass"])
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title("Confusion Matrix - Student Performance", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Confusion matrix plot saved: {output_path}")
    return output_path

def plot_feature_importance(model, feature_names, output_path=None):
    """
    Generate and save feature importance horizontal bar plot.
    """
    if output_path is None:
        output_path = os.path.join(MODELS_DIR, "feature_importance.png")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("viridis", len(sorted_features))
    sns.barplot(
        x=sorted_importances,
        y=sorted_features,
        hue=sorted_features,
        legend=False,
        palette=palette,
        ax=ax
    )
    
    ax.set_title("RandomForest Feature Importance", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[+] Feature importance plot saved: {output_path}")
    return output_path

def train_and_log(
    data_path: str = "data/student_data.csv",
    run_name: str = "RandomForest_Baseline_Training",
    n_estimators: int = 100,
    random_state: int = 42,
    max_depth: int = 10,
    save_local_model: bool = True
):
    """
    Train RandomForestClassifier and track experiment using MLflow.
    """
    print(f"\n=======================================================")
    print(f" Starting Training Run: {run_name}")
    print(f" Dataset: {data_path}")
    print(f"=======================================================")
    
    # 1. Resolve absolute file path for data
    if not os.path.isabs(data_path):
        resolved_data_path = os.path.join(BASE_DIR, data_path)
    else:
        resolved_data_path = data_path
        
    # 2. Set MLflow tracking URI and Experiment
    tracking_uri = f"file:///{MLRUNS_DIR.replace(os.sep, '/')}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 3. Data Preparation
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(
        filepath=resolved_data_path,
        test_size=0.2,
        random_state=random_state,
        fit_preprocessor=True
    )
    
    print(f"[*] Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"[*] Features ({len(feature_names)}): {feature_names}")
    
    # 4. Model Initialization
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight="balanced"
    )
    
    # 5. Train Model
    model.fit(X_train, y_train)
    
    # 6. Evaluate Model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    
    print("\n---------------- Model Evaluation Metrics ----------------")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print("----------------------------------------------------------\n")
    
    # 7. Generate Visual Artifacts
    cm_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    fi_path = os.path.join(MODELS_DIR, "feature_importance.png")
    plot_confusion_matrix(y_test, y_pred, output_path=cm_path)
    plot_feature_importance(model, feature_names, output_path=fi_path)
    
    # 8. MLflow Logging
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[+] Logging run to MLflow (Run ID: {run_id})")
        
        # Log Parameters
        mlflow.log_param("model_name", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_source", data_path)
        
        # Log Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log Artifacts
        mlflow.log_artifact(cm_path, artifact_path="evaluation_plots")
        mlflow.log_artifact(fi_path, artifact_path="evaluation_plots")
        
        preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
        if os.path.exists(preprocessor_path):
            mlflow.log_artifact(preprocessor_path, artifact_path="preprocessor")
            
        # Log Model via MLflow sklearn module with cloudpickle serialization
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            serialization_format="cloudpickle"
        )
        
        print(f"[+] Run '{run_name}' successfully logged to MLflow under '{EXPERIMENT_NAME}'.")
        
    # 9. Save local production model for standalone deployment
    if save_local_model:
        model_file = os.path.join(MODELS_DIR, "student_model.joblib")
        joblib.dump(model, model_file)
        
        # Store metadata for API
        metadata = {
            "run_id": run_id,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "n_estimators": n_estimators,
            "feature_names": feature_names
        }
        joblib.dump(metadata, os.path.join(MODELS_DIR, "model_metadata.joblib"))
        print(f"[+] Serialized model and metadata saved to: {model_file}")

    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "model": model
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Student Performance Prediction Model")
    parser.add_argument("--data-path", type=str, default="data/student_data.csv", help="Path to input dataset")
    parser.add_argument("--run-name", type=str, default="RandomForest_Baseline_Training", help="MLflow Run Name")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees in RandomForest")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--max-depth", type=int, default=10, help="Maximum tree depth")
    
    args = parser.parse_args()
    train_and_log(
        data_path=args.data_path,
        run_name=args.run_name,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        max_depth=args.max_depth
    )

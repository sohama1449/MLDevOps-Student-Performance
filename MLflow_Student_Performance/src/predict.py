"""
Inference Module and CLI for Student Performance Prediction.
Loads the serialized production model and preprocessor to generate predictions.
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

BASE_DIR = os.path.dirname(CURRENT_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import joblib
import pandas as pd
import numpy as np

from data_preprocessing import (
    clean_data,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)

MODEL_PATH = os.path.join(BASE_DIR, "models", "student_model.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "models", "model_metadata.joblib")

# Default sensible fallback values if optional features are omitted in API calls
DEFAULT_INPUTS = {
    "gender": "Female",
    "parental_education": "Bachelor",
    "internet_access": "Yes"
}

_model = None
_preprocessor = None

def get_model_and_preprocessor():
    """
    Singleton loader for model and preprocessor to optimize inference latency.
    """
    global _model, _preprocessor
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run training first.")
        _model = joblib.load(MODEL_PATH)
        
    if _preprocessor is None:
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor not found at: {PREPROCESSOR_PATH}. Run training first.")
        _preprocessor = joblib.load(PREPROCESSOR_PATH)
        
    return _model, _preprocessor

def predict_single(input_data: dict) -> dict:
    """
    Perform prediction for a single student profile.
    
    Args:
        input_data (dict): Dictionary with keys like 'study_time', 'absences', 'previous_score', etc.
        
    Returns:
        dict: {"prediction": "Pass"|"Fail", "confidence": float}
    """
    model, preprocessor = get_model_and_preprocessor()
    
    # Merge with defaults
    full_data = DEFAULT_INPUTS.copy()
    full_data.update(input_data)
    
    # Construct DataFrame
    df = pd.DataFrame([full_data])
    df = clean_data(df)
    
    # Transform features
    X_trans = preprocessor.transform(df[FEATURE_COLUMNS])
    
    # Predict probabilities
    probabilities = model.predict_proba(X_trans)[0]
    # Class 0: Fail, Class 1: Pass
    pass_prob = float(probabilities[1])
    
    if pass_prob >= 0.50:
        prediction = "Pass"
        confidence = float(np.round(pass_prob, 2))
    else:
        prediction = "Fail"
        confidence = float(np.round(1.0 - pass_prob, 2))
        
    return {
        "prediction": prediction,
        "confidence": confidence
    }

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform batch inference on a DataFrame of student records.
    """
    model, preprocessor = get_model_and_preprocessor()
    
    df_clean = clean_data(df)
    for col, default_val in DEFAULT_INPUTS.items():
        if col not in df_clean.columns:
            df_clean[col] = default_val
            
    X_trans = preprocessor.transform(df_clean[FEATURE_COLUMNS])
    probs = model.predict_proba(X_trans)
    
    pass_probs = probs[:, 1]
    predictions = np.where(pass_probs >= 0.50, "Pass", "Fail")
    confidences = np.where(pass_probs >= 0.50, pass_probs, 1.0 - pass_probs).round(2)
    
    result_df = df.copy()
    result_df["prediction"] = predictions
    result_df["confidence"] = confidences
    return result_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Performance Prediction CLI")
    parser.add_argument("--study-time", type=float, default=5.0, help="Weekly study hours")
    parser.add_argument("--absences", type=int, default=2, help="Number of absences")
    parser.add_argument("--previous-score", type=float, default=75.0, help="Previous exam score (0-100)")
    parser.add_argument("--gender", type=str, default="Female", help="Gender: Male/Female")
    parser.add_argument("--parental-education", type=str, default="Bachelor", help="Parental Education")
    parser.add_argument("--internet-access", type=str, default="Yes", help="Internet Access: Yes/No")
    
    args = parser.parse_args()
    
    sample_input = {
        "study_time": args.study_time,
        "absences": args.absences,
        "previous_score": args.previous_score,
        "gender": args.gender,
        "parental_education": args.parental_education,
        "internet_access": args.internet_access
    }
    
    result = predict_single(sample_input)
    print("\n---------------- Prediction Result ----------------")
    print(f"  Input       : {sample_input}")
    print(f"  Prediction  : {result['prediction']}")
    print(f"  Confidence  : {result['confidence']}")
    print("---------------------------------------------------\n")

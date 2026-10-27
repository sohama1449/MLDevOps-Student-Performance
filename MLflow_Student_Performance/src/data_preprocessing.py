"""
Data Preprocessing Module for Student Performance Prediction.
Handles data loading, cleaning, encoding, transformations, and train-test splitting.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define schema constants
NUMERICAL_FEATURES = ["study_time", "absences", "previous_score"]
CATEGORICAL_FEATURES = ["gender", "parental_education", "internet_access"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "passed"

# Expected categories for categorical features to guarantee deterministic encoding
CATEGORY_MAPPINGS = {
    "gender": ["Female", "Male"],
    "parental_education": ["None", "High School", "Bachelor", "Master", "PhD"],
    "internet_access": ["No", "Yes"]
}

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV dataset.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at: {filepath}")
    df = pd.read_csv(filepath)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data hygiene and cleaning:
    - Strip whitespace
    - Impute missing values (median for numerical, mode for categorical)
    - Enforce bounds on numerical attributes
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    
    # Strip whitespace from string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        
    # Handle missing numerical features with median imputation
    for num_col in NUMERICAL_FEATURES:
        if num_col in df.columns:
            if df[num_col].isnull().sum() > 0:
                df[num_col] = df[num_col].fillna(df[num_col].median())
            # Enforce non-negative bounds
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce").fillna(0)
            
    # Clip numerical ranges to realistic physical bounds
    if "study_time" in df.columns:
        df["study_time"] = df["study_time"].clip(0.0, 30.0)
    if "absences" in df.columns:
        df["absences"] = df["absences"].clip(0, 100)
    if "previous_score" in df.columns:
        df["previous_score"] = df["previous_score"].clip(0.0, 100.0)
        
    # Handle missing categorical features with sensible mode imputation
    for cat_col in CATEGORICAL_FEATURES:
        if cat_col in df.columns:
            if df[cat_col].isnull().sum() > 0:
                df[cat_col] = df[cat_col].fillna(CATEGORY_MAPPINGS[cat_col][0])
            # If unexpected category appears, map to first category
            valid_cats = CATEGORY_MAPPINGS[cat_col]
            df[cat_col] = df[cat_col].apply(lambda x: x if x in valid_cats else valid_cats[0])
            
    # Standardize target column if present
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(
            lambda x: "Pass" if str(x).strip().lower() in ["pass", "1", "true", "p"] else "Fail"
        )
        
    return df

def get_preprocessor():
    """
    Create a scikit-learn ColumnTransformer for numerical scaling and one-hot encoding.
    
    Returns:
        ColumnTransformer: Preprocessing transformer pipeline.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(
        categories=[CATEGORY_MAPPINGS[col] for col in CATEGORICAL_FEATURES],
        handle_unknown="ignore",
        sparse_output=False
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop"
    )
    return preprocessor

def fit_and_save_preprocessor(df: pd.DataFrame, save_path: str = "models/preprocessor.joblib"):
    """
    Fit the preprocessor on the dataset features and persist it using joblib.
    
    Args:
        df (pd.DataFrame): Data containing feature columns.
        save_path (str): Destination file path for serialized preprocessor.
        
    Returns:
        ColumnTransformer: Fitted preprocessor.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    preprocessor = get_preprocessor()
    preprocessor.fit(df[FEATURE_COLUMNS])
    joblib.dump(preprocessor, save_path)
    print(f"[+] Preprocessor fitted and saved to: {save_path}")
    return preprocessor

def load_preprocessor(load_path: str = "models/preprocessor.joblib"):
    """
    Load saved preprocessor artifact from disk.
    
    Args:
        load_path (str): Path to serialized preprocessor.
        
    Returns:
        ColumnTransformer: Fitted preprocessor object.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Preprocessor not found at: {load_path}")
    return joblib.load(load_path)

def prepare_data(filepath: str, test_size: float = 0.2, random_state: int = 42, fit_preprocessor: bool = True):
    """
    End-to-end data preparation workflow:
    1. Loads dataset
    2. Cleans records
    3. Fits/loads preprocessor
    4. Encodes features and binary target
    5. Splits into train and test sets
    
    Args:
        filepath (str): Path to CSV data.
        test_size (float): Proportion of test data.
        random_state (int): Random seed.
        fit_preprocessor (bool): Whether to fit a new preprocessor or load existing.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    df = load_data(filepath)
    df = clean_data(df)
    
    if fit_preprocessor:
        preprocessor = fit_and_save_preprocessor(df)
    else:
        preprocessor = load_preprocessor()
        
    X_transformed = preprocessor.transform(df[FEATURE_COLUMNS])
    
    # Generate human-readable feature names from transformer
    cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names = NUMERICAL_FEATURES + list(cat_feature_names)
    
    # Binary target encoding: 1 for Pass, 0 for Fail
    y = (df[TARGET_COLUMN] == "Pass").astype(int).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor, feature_names

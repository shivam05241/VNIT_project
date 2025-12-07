"""
Data Preprocessing and EDA Module
==================================

This module contains functions for loading, exploring, and preprocessing
data for the house price prediction model, orchestrated with Prefect.

It includes:
- Data loading and basic exploration
- Exploratory Data Analysis (EDA) with visualizations
- Feature selection based on correlation
- Data preprocessing (encoding, scaling)

Prefect Integration:
- Tasks are decorated with @task for workflow tracking
- Main flow is decorated with @flow for orchestration
- Run with: python preprocessing.py
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
import optuna

try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    # Fallback decorators if Prefect is not available
    def flow(name=None, description=None):
        def decorator(func):
            return func
        return decorator
    
    def task(name=None, description=None):
        def decorator(func):
            return func
        return decorator

# Evidently for data quality monitoring
try:
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset, DataDriftPreset
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


@task(name="Load Data", description="Load training data from CSV file")
def load_data(filepath):
    """
    Load data from a CSV file.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file to load.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    df = pd.read_csv(filepath)
    print(f"✓ Loaded data from {filepath} with shape {df.shape}")
    return df


@task(name="Basic EDA", description="Perform basic exploratory data analysis")
def basic_eda(df):
    """
    Perform basic exploratory data analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    """
    print(f"Dataset Shape: {df.shape}")
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe().T)
    print("✓ Basic EDA completed")


@task(name="Plot Correlations", description="Plot correlation heatmap for numeric columns")
def plot_correlation_heatmap(df):
    """
    Plot a correlation heatmap for numeric columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing numeric columns.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="RdBu")
    plt.title("Correlations Between Variables", size=15)
    plt.savefig("correlation_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("✓ Correlation heatmap saved as correlation_heatmap.png")


@task(name="Select Features", description="Select important features based on correlation threshold")
def select_important_features(df, corr_threshold=0.50):
    """
    Select features based on correlation with target variable (SalePrice).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numeric columns and 'SalePrice' target.
    corr_threshold : float, optional
        Correlation threshold for feature selection (default: 0.50).
    
    Returns
    -------
    tuple
        (important_num_cols, cat_cols, important_cols, df_filtered)
        - important_num_cols: list of important numeric column names
        - cat_cols: list of categorical column names
        - important_cols: combined list of important features
        - df_filtered: filtered dataframe with only important columns
    """
    # Select numeric columns with high correlation to SalePrice
    corr_with_target = df.select_dtypes(include=np.number).corr()["SalePrice"]
    important_num_cols = list(
        corr_with_target[
            (corr_with_target > corr_threshold) | (corr_with_target < -corr_threshold)
        ].index
    )
    
    # Define categorical columns
    cat_cols = [
        "MSZoning",
        "Utilities",
        "BldgType",
        "Heating",
        "KitchenQual",
        "SaleCondition",
        "LandSlope",
    ]
    
    # Combine important columns
    important_cols = important_num_cols + cat_cols
    
    # Filter dataframe
    df_filtered = df[important_cols].copy()
    
    print(f"✓ Selected {len(important_cols)} important features")
    return important_num_cols, cat_cols, important_cols, df_filtered


@task(name="Data Quality Check", description="Generate Evidently data quality report")
def check_data_quality(df, name="Data Quality Report"):
    """
    Generate Evidently data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to analyze.
    name : str, optional
        Name of the report (default: "Data Quality Report").
    
    Returns
    -------
    dict or None
        Report summary or None if Evidently not available.
    """
    if not EVIDENTLY_AVAILABLE:
        print("⚠ Evidently not available - skipping data quality check")
        return None
    
    try:
        report = Report(metrics=[DataQualityPreset()])
        report.run(reference_data=df)
        
        # Save report
        report.save_html("data_quality_report.html")
        print("✓ Data quality report generated: data_quality_report.html")
        
        return {"status": "completed", "report_file": "data_quality_report.html"}
    except Exception as e:
        print(f"⚠ Error generating quality report: {e}")
        return None


@task(name="Check Missing Values", description="Report missing values in dataframe")
def check_missing_values(df):
    """
    Check and report missing values in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to check for missing values.
    """
    missing_sum = df.isna().sum().sum()
    print("Missing Values by Column")
    print("-" * 30)
    print(df.isna().sum())
    print("-" * 30)
    print("TOTAL MISSING VALUES:", missing_sum)
    print("✓ Missing values check completed")
    """
    Check and report missing values in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to check for missing values.
    """
    missing_sum = df.isna().sum().sum()
    print("Missing Values by Column")
    print("-" * 30)
    print(df.isna().sum())
    print("-" * 30)
    print("TOTAL MISSING VALUES:", missing_sum)
    print("✓ Missing values check completed")


@task(name="Preprocess Data", description="Encode categorical variables and scale numeric features")
def preprocess_data(df, cat_cols, important_num_cols, target_col="SalePrice"):
    """
    Preprocess data by encoding categorical variables and scaling numeric features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to preprocess.
    cat_cols : list
        List of categorical column names.
    important_num_cols : list
        List of important numeric column names.
    target_col : str, optional
        Name of the target column (default: "SalePrice").
    
    Returns
    -------
    tuple
        (X, y, scaler)
        - X: preprocessed feature dataframe
        - y: target variable series
        - scaler: fitted StandardScaler for numeric features
    """
    # Separate target from features
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode categorical variables
    X = pd.get_dummies(X, columns=cat_cols)
    
    # Remove target from numeric columns if present
    numeric_cols_to_scale = [col for col in important_num_cols if col != target_col]
    
    # Scale numeric features
    scaler = StandardScaler()
    X[numeric_cols_to_scale] = scaler.fit_transform(X[numeric_cols_to_scale])
    
    print(f"✓ Preprocessing complete - Features: {X.shape}, Target: {y.shape}")
    return X, y, scaler


@flow(name="Data Preprocessing Pipeline", description="End-to-end data preprocessing workflow")
def main(data_path="../data/train.csv"):
    """
    Main Prefect flow: load, explore, and preprocess data.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the training data CSV file.
    
    Returns
    -------
    tuple
        (X, y, scaler, important_cols)
        - X: preprocessed features
        - y: target variable
        - scaler: fitted scaler for numeric features
        - important_cols: list of important column names
    """
    print("\n" + "=" * 60)
    print("PREFECT DATA PREPROCESSING PIPELINE")
    print("=" * 60 + "\n")
    
    # Load data
    df = load_data(data_path)
    
    # Perform EDA
    print("\n" + "=" * 60)
    print("STAGE 1: EXPLORATORY DATA ANALYSIS")
    print("=" * 60 + "\n")
    basic_eda(df)
    
    # Plot correlations
    print("\n" + "=" * 60)
    print("STAGE 2: CORRELATION ANALYSIS")
    print("=" * 60 + "\n")
    plot_correlation_heatmap(df)
    
    # Select important features
    print("\n" + "=" * 60)
    print("STAGE 3: FEATURE SELECTION")
    print("=" * 60 + "\n")
    important_num_cols, cat_cols, important_cols, df_filtered = select_important_features(df)
    print(f"Important Features: {important_cols}\n")
    
    # Check missing values
    print("\n" + "=" * 60)
    print("STAGE 4: MISSING VALUES CHECK")
    print("=" * 60 + "\n")
    check_missing_values(df_filtered)
    
    # Preprocess data
    print("\n" + "=" * 60)
    print("STAGE 5: DATA PREPROCESSING")
    print("=" * 60 + "\n")
    X, y, scaler = preprocess_data(df_filtered, cat_cols, important_num_cols)
    
    print("\nFirst few rows of preprocessed features:")
    print(X.head())
    
    print("\n" + "=" * 60)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")
    
    return X, y, scaler, important_cols


if __name__ == "__main__":
    # Run the preprocessing pipeline with Prefect
    # This will execute the flow and log all task completions
    import sys
    
    data_path = sys.argv[1] if len(sys.argv) > 1 else "../data/train.csv"
    X, y, scaler, important_cols = main(data_path)

"""
Model Training and Optimization Module
======================================

This module implements XGBoost model training with Optuna hyperparameter optimization,
MLflow experiment tracking, and Evidently model performance monitoring.

Features:
- Hyperparameter optimization using Optuna
- Model training and evaluation
- MLflow experiment tracking
- Performance monitoring with Evidently
- Model persistence and versioning
"""

import numpy as np
import pandas as pd
import warnings
import os
from datetime import datetime
import json

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import optuna
import joblib
import mlflow
import mlflow.sklearn

try:
    from prefect import flow, task
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(name=None, description=None):
        def decorator(func):
            return func
        return decorator
    
    def task(name=None, description=None):
        def decorator(func):
            return func
        return decorator


# ============================================================================
# TASK: Data Splitting
# ============================================================================
@task(name="Split Data", description="Split data into train and test sets")
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Proportion of data for testing (default: 0.2).
    random_state : int, optional
        Random state for reproducibility (default: 42).
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✓ Data split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train, X_test, y_train, y_test


# ============================================================================
# TASK: Model Evaluation
# ============================================================================
@task(name="Evaluate Model", description="Calculate model performance metrics")
def evaluation(y_true, y_pred):
    """
    Calculate evaluation metrics for regression model.
    
    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    
    Returns
    -------
    dict
        Dictionary containing MAE, MSE, RMSE, and R² metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y_true, y_pred)
    
    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r_squared),
    }
    
    print(f"✓ Model Evaluation:")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r_squared:.4f}")
    
    return metrics


# ============================================================================
# TASK: Optuna Hyperparameter Optimization
# ============================================================================
@task(name="Hyperparameter Optimization", description="Optimize XGBoost parameters using Optuna")
def optimize_hyperparameters(X, y, n_trials=50, cv=5):
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_trials : int, optional
        Number of optimization trials (default: 50).
    cv : int, optional
        Cross-validation folds (default: 5).
    
    Returns
    -------
    dict
        Best parameters found by Optuna.
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.4),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
            "objective": "reg:squarederror",
        }
        
        model = XGBRegressor(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(
            model, X, y,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1
        )
        
        return np.sqrt(-score.mean())
    
    print(f"Starting Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_params = study.best_params
    best_rmse = study.best_value
    
    print(f"✓ Optimization complete!")
    print(f"  Best RMSE: {best_rmse:.4f}")
    print(f"  Best Params: {best_params}")
    
    return {
        "best_params": best_params,
        "best_rmse": float(best_rmse),
        "study": study,
    }


# ============================================================================
# TASK: Model Training
# ============================================================================
@task(name="Train Model", description="Train XGBoost model with best parameters")
def train_model(X_train, X_test, y_train, y_test, best_params):
    """
    Train XGBoost model with optimized parameters.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    y_train : pd.Series
        Training target.
    y_test : pd.Series
        Testing target.
    best_params : dict
        Best parameters from optimization.
    
    Returns
    -------
    XGBRegressor
        Trained model.
    """
    params = best_params.copy()
    params["objective"] = "reg:squarederror"
    
    model = XGBRegressor(**params, random_state=42)
    
    print("Training model with best parameters...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    
    print(f"✓ Model training complete!")
    return model


# ============================================================================
# TASK: Predictions
# ============================================================================
@task(name="Generate Predictions", description="Generate predictions on test set")
def generate_predictions(model, X_test):
    """
    Generate predictions using trained model.
    
    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    X_test : pd.DataFrame
        Test features.
    
    Returns
    -------
    np.ndarray
        Predictions.
    """
    predictions = model.predict(X_test)
    print(f"✓ Generated {len(predictions)} predictions")
    return predictions


# ============================================================================
# TASK: Feature Importance
# ============================================================================
@task(name="Feature Importance", description="Extract feature importance from model")
def get_feature_importance(model, top_n=10):
    """
    Get top N important features from model.
    
    Parameters
    ----------
    model : XGBRegressor
        Trained XGBoost model.
    top_n : int, optional
        Number of top features to return (default: 10).
    
    Returns
    -------
    pd.DataFrame
        Top N features with importance scores.
    """
    feature_imp = pd.DataFrame({
        "feature": model.feature_names_in_,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"✓ Top {top_n} Important Features:")
    print(feature_imp.head(top_n).to_string(index=False))
    
    return feature_imp


# ============================================================================
# TASK: Model Persistence
# ============================================================================
@task(name="Save Model", description="Save trained model to disk")
def save_model(model, X_train, model_path="../../models/xgboost_model.joblib",
               data_path="../../data/X_train.csv"):
    """
    Save trained model and training data to disk.
    
    Parameters
    ----------
    model : XGBRegressor
        Trained model.
    X_train : pd.DataFrame
        Training features.
    model_path : str, optional
        Path to save model (default: "../../models/xgboost_model.joblib").
    data_path : str, optional
        Path to save training data (default: "../../data/X_train.csv").
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✓ Model saved to {model_path}")
    
    # Save training data
    X_train.to_csv(data_path, index=False)
    print(f"✓ Training data saved to {data_path}")


# ============================================================================
# TASK: MLflow Logging
# ============================================================================
@task(name="MLflow Logging", description="Log model, metrics, and params to MLflow")
def log_to_mlflow(best_params, metrics, best_rmse, X, X_train, X_test, 
                  model, important_cols=None):
    """
    Log model training details to MLflow.
    
    Parameters
    ----------
    best_params : dict
        Best hyperparameters.
    metrics : dict
        Evaluation metrics.
    best_rmse : float
        Best RMSE from Optuna.
    X : pd.DataFrame
        Full feature matrix.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Testing features.
    model : XGBRegressor
        Trained model.
    important_cols : list, optional
        Important features used.
    """
    with mlflow.start_run(run_name="xgb_optuna_run") as run:
        # Log parameters
        mlflow.log_params({
            k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
            for k, v in best_params.items()
        })
        
        # Log metadata
        mlflow.log_param("n_features", int(X.shape[1]))
        mlflow.log_param("n_train", int(X_train.shape[0]))
        mlflow.log_param("n_test", int(X_test.shape[0]))
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        mlflow.log_metric("optuna_best_rmse", best_rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, artifact_path="final_xgb_model")
        
        # Log important columns
        if important_cols:
            mlflow.log_param("important_cols", ",".join(important_cols))
        
        # Log tags
        mlflow.set_tag("model_framework", "xgboost")
        mlflow.set_tag("tuner", "optuna")
        
        print(f"✓ MLflow run created: {run.info.run_id}")
        return run.info.run_id


# ============================================================================
# TASK: Evidently Model Performance Report
# ============================================================================
@task(name="Performance Report", description="Generate Evidently performance report")
def generate_performance_report(y_test, y_pred, output_dir="reports", name="model_performance"):
    """
    Generate model performance report.
    
    Parameters
    ----------
    y_test : pd.Series
        True test values.
    y_pred : np.ndarray
        Predicted values.
    output_dir : str, optional
        Output directory for reports (default: "reports").
    name : str, optional
        Report name prefix (default: "model_performance").
    
    Returns
    -------
    dict
        Report metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate residuals
    residuals = y_test.values - y_pred
    
    # Create performance summary
    performance_report = {
        "timestamp": timestamp,
        "predictions_count": len(y_pred),
        "residual_stats": {
            "mean": float(np.mean(residuals)),
            "std": float(np.std(residuals)),
            "min": float(np.min(residuals)),
            "max": float(np.max(residuals)),
            "median": float(np.median(residuals)),
        },
        "prediction_stats": {
            "mean": float(np.mean(y_pred)),
            "std": float(np.std(y_pred)),
            "min": float(np.min(y_pred)),
            "max": float(np.max(y_pred)),
        },
        "actual_stats": {
            "mean": float(y_test.mean()),
            "std": float(y_test.std()),
            "min": float(y_test.min()),
            "max": float(y_test.max()),
        },
    }
    
    # Save report
    report_file = os.path.join(output_dir, f"{name}_{timestamp}.json")
    with open(report_file, "w") as f:
        json.dump(performance_report, f, indent=2)
    
    print(f"✓ Performance report saved: {report_file}")
    
    return {
        "status": "success",
        "file": report_file,
        "report": performance_report,
    }


# ============================================================================
# MAIN FLOW
# ============================================================================
@flow(name="Model Training Pipeline", description="End-to-end model training with optimization and monitoring")
def main(X, y, n_trials=50, important_cols=None):
    """
    Main Prefect flow for model training pipeline.
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    n_trials : int, optional
        Number of Optuna trials (default: 50).
    important_cols : list, optional
        Important features used in preprocessing.
    
    Returns
    -------
    dict
        Pipeline results including model, metrics, and artifacts.
    """
    print("\n" + "=" * 70)
    print("MODEL TRAINING PIPELINE WITH PREFECT & EVIDENTLY")
    print("=" * 70 + "\n")
    
    # Stage 1: Data Split
    print("STAGE 1: DATA SPLIT")
    print("-" * 70)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print()
    
    # Stage 2: Hyperparameter Optimization
    print("STAGE 2: HYPERPARAMETER OPTIMIZATION")
    print("-" * 70)
    optuna_result = optimize_hyperparameters(X, y, n_trials=n_trials)
    best_params = optuna_result["best_params"]
    best_rmse = optuna_result["best_rmse"]
    print()
    
    # Stage 3: Model Training
    print("STAGE 3: MODEL TRAINING")
    print("-" * 70)
    model = train_model(X_train, X_test, y_train, y_test, best_params)
    print()
    
    # Stage 4: Predictions
    print("STAGE 4: PREDICTIONS")
    print("-" * 70)
    y_pred = generate_predictions(model, X_test)
    print()
    
    # Stage 5: Model Evaluation
    print("STAGE 5: MODEL EVALUATION")
    print("-" * 70)
    metrics = evaluation(y_test, y_pred)
    print()
    
    # Stage 6: Feature Importance
    print("STAGE 6: FEATURE IMPORTANCE")
    print("-" * 70)
    feature_importance = get_feature_importance(model, top_n=10)
    print()
    
    # Stage 7: Model Persistence
    print("STAGE 7: MODEL PERSISTENCE")
    print("-" * 70)
    save_model(model, X_train)
    print()
    
    # Stage 8: MLflow Logging
    print("STAGE 8: MLFLOW LOGGING")
    print("-" * 70)
    mlflow_run_id = log_to_mlflow(best_params, metrics, best_rmse, X, 
                                  X_train, X_test, model, important_cols)
    print()
    
    # Stage 9: Performance Monitoring (Evidently)
    print("STAGE 9: PERFORMANCE MONITORING")
    print("-" * 70)
    perf_report = generate_performance_report(y_test, y_pred)
    print()
    
    # Summary
    print("=" * 70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Model R² Score: {metrics['r2']:.4f}")
    print(f"Model RMSE: {metrics['rmse']:.4f}")
    print(f"MLflow Run ID: {mlflow_run_id}")
    print("=" * 70 + "\n")
    
    return {
        "model": model,
        "metrics": metrics,
        "best_params": best_params,
        "feature_importance": feature_importance,
        "mlflow_run_id": mlflow_run_id,
        "performance_report": perf_report,
    }


if __name__ == "__main__":
    # Import preprocessing module
    import sys
    sys.path.insert(0, "/home/shivam/workspace/VNIT_project/src/preprocessing")
    
    from preprocessing import main as preprocess_main
    
    # Run preprocessing pipeline
    print("Loading and preprocessing data...")
    X, y, scaler, important_cols = preprocess_main("../../data/train.csv")
    
    # Run modeling pipeline
    results = main(X, y, n_trials=50, important_cols=important_cols)
    
    print("\n✓ All pipelines completed successfully!")

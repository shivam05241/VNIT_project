.. _module-modeling:

===============
Modeling Module
===============

The modeling module implements hyperparameter optimization, model training, evaluation, and MLflow experiment tracking.

Overview
========

The modeling pipeline consists of 9 stages orchestrated with Prefect:

1. **Data Splitting** - Train/test split
2. **Hyperparameter Optimization** - Optuna-based tuning
3. **Model Training** - XGBoost with best parameters
4. **Predictions** - Generate test predictions
5. **Evaluation** - Calculate metrics
6. **Feature Importance** - Extract feature importance
7. **Model Persistence** - Save model to disk
8. **MLflow Logging** - Track experiment
9. **Performance Monitoring** - Evidently reports

Module Functions
================

split_data(X, y, test_size=0.2, random_state=42)
------------------------------------------------

Split data into training and testing sets.

**Parameters:**

- ``X`` (pd.DataFrame): Feature matrix
- ``y`` (pd.Series): Target variable
- ``test_size`` (float): Test set proportion (default: 0.2)
- ``random_state`` (int): Reproducibility seed (default: 42)

**Returns:**

- ``tuple``: (X_train, X_test, y_train, y_test)

**Example:**

::

    from modeling import split_data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

evaluation(y_true, y_pred)
--------------------------

Calculate regression evaluation metrics.

**Parameters:**

- ``y_true`` (array-like): True values
- ``y_pred`` (array-like): Predicted values

**Returns:**

- ``dict``: Metrics (MAE, MSE, RMSE, R²)

**Metrics Explained:**

- **MAE** (Mean Absolute Error): Average absolute prediction error
- **MSE** (Mean Squared Error): Average squared prediction error
- **RMSE** (Root Mean Squared Error): √MSE
- **R²** (R-squared): Proportion of variance explained (0-1)

**Example:**

::

    from modeling import evaluation
    metrics = evaluation(y_test, predictions)
    print(f"R²: {metrics['r2']:.4f}")
    print(f"RMSE: {metrics['rmse']:.2f}")

optimize_hyperparameters(X, y, n_trials=50, cv=5)
-------------------------------------------------

Optimize XGBoost hyperparameters using Optuna.

**Parameters:**

- ``X`` (pd.DataFrame): Feature matrix
- ``y`` (pd.Series): Target variable
- ``n_trials`` (int): Number of trials (default: 50)
- ``cv`` (int): Cross-validation folds (default: 5)

**Returns:**

- ``dict``: Optimization results (best_params, best_rmse, study)

**Optimized Parameters:**

- ``n_estimators``: 500-2000 (number of trees)
- ``learning_rate``: 0.01-0.2 (step size)
- ``max_depth``: 3-8 (tree depth)
- ``min_child_weight``: 1-7 (minimum leaf samples)
- ``subsample``: 0.6-1.0 (row sampling ratio)
- ``colsample_bytree``: 0.6-1.0 (column sampling ratio)
- ``gamma``: 0-0.4 (regularization)
- ``reg_alpha``: 0.0-0.1 (L1 regularization)
- ``reg_lambda``: 0.5-2.0 (L2 regularization)

**Example:**

::

    from modeling import optimize_hyperparameters
    
    opt_result = optimize_hyperparameters(X, y, n_trials=100)
    best_params = opt_result['best_params']
    best_rmse = opt_result['best_rmse']
    print(f"Best RMSE: {best_rmse:.2f}")

train_model(X_train, X_test, y_train, y_test, best_params)
----------------------------------------------------------

Train XGBoost with optimized parameters.

**Parameters:**

- ``X_train``, ``X_test``, ``y_train``, ``y_test``: Train/test data
- ``best_params`` (dict): Best parameters from optimization

**Returns:**

- ``XGBRegressor``: Trained model

**Example:**

::

    from modeling import train_model
    model = train_model(X_train, X_test, y_train, y_test, best_params)

generate_predictions(model, X_test)
-----------------------------------

Generate predictions on test set.

**Parameters:**

- ``model`` (XGBRegressor): Trained model
- ``X_test`` (pd.DataFrame): Test features

**Returns:**

- ``np.ndarray``: Predictions

**Example:**

::

    from modeling import generate_predictions
    predictions = generate_predictions(model, X_test)

get_feature_importance(model, top_n=10)
---------------------------------------

Extract top N important features.

**Parameters:**

- ``model`` (XGBRegressor): Trained model
- ``top_n`` (int): Number of features to return (default: 10)

**Returns:**

- ``pd.DataFrame``: Feature importance dataframe

**Example:**

::

    from modeling import get_feature_importance
    importance = get_feature_importance(model, top_n=15)
    print(importance)

save_model(model, X_train, model_path, data_path)
-------------------------------------------------

Save model and training data to disk.

**Parameters:**

- ``model`` (XGBRegressor): Trained model
- ``X_train`` (pd.DataFrame): Training features
- ``model_path`` (str): Path to save model
- ``data_path`` (str): Path to save training data

**Example:**

::

    from modeling import save_model
    save_model(model, X_train)

log_to_mlflow(best_params, metrics, best_rmse, X, X_train, X_test, model)
-------------------------------------------------------------------------

Log experiment to MLflow.

**Parameters:**

- ``best_params`` (dict): Best hyperparameters
- ``metrics`` (dict): Evaluation metrics
- ``best_rmse`` (float): Best RMSE from optimization
- ``X``, ``X_train``, ``X_test``: Data splits
- ``model`` (XGBRegressor): Trained model

**Example:**

::

    from modeling import log_to_mlflow
    run_id = log_to_mlflow(best_params, metrics, best_rmse, X, X_train, X_test, model)
    print(f"MLflow Run ID: {run_id}")

generate_performance_report(y_test, y_pred, output_dir, name)
-------------------------------------------------------------

Generate Evidently performance report.

**Parameters:**

- ``y_test`` (pd.Series): True test values
- ``y_pred`` (np.ndarray): Predictions
- ``output_dir`` (str): Output directory
- ``name`` (str): Report name prefix

**Returns:**

- ``dict``: Report metadata

**Example:**

::

    from modeling import generate_performance_report
    report = generate_performance_report(y_test, predictions)

main(X, y, n_trials=50, important_cols=None)
--------------------------------------------

Run complete modeling pipeline with Prefect orchestration.

**Parameters:**

- ``X`` (pd.DataFrame): Feature matrix
- ``y`` (pd.Series): Target variable
- ``n_trials`` (int): Optuna optimization trials (default: 50)
- ``important_cols`` (list): Important column names (optional)

**Returns:**

- ``dict``: Pipeline results

**Example:**

::

    from modeling import main
    results = main(X, y, n_trials=100)

Model Performance
=================

Standard test set performance:

- **R² Score**: 0.8990
- **RMSE**: $27,836.04
- **MAE**: $17,279.64

Hyperparameter Tuning
=====================

Optuna performs intelligent search across parameter space:

- **Pruning**: Stops unpromising trials early
- **Sampling**: Uses TPE (Tree-structured Parzen Estimator)
- **Parallelization**: Can use multiple cores

Cross-Validation
================

5-fold cross-validation used for:

- Parameter optimization
- Model evaluation
- Robustness assessment

Usage Example
=============

Complete modeling pipeline::

    from preprocessing import main as preprocess
    from modeling import main as train_model
    
    # Preprocess data
    X, y, scaler, cols = preprocess("data/train.csv")
    
    # Train model
    results = train_model(X, y, n_trials=50, important_cols=cols)
    
    # Access results
    model = results['model']
    metrics = results['metrics']
    importance = results['feature_importance']
    mlflow_id = results['mlflow_run_id']
    
    print(f"Final R²: {metrics['r2']:.4f}")

Troubleshooting
===============

**Issue**: Optuna optimization too slow

**Solution**: Reduce n_trials or use GPU::

    results = optimize_hyperparameters(X, y, n_trials=20)

**Issue**: Out of memory during training

**Solution**: Use early stopping or reduce dataset::

    model.fit(X_train, y_train, early_stopping_rounds=10)

**Issue**: Model overfitting

**Solution**: Increase regularization parameters::

    best_params['reg_alpha'] = 0.5
    best_params['reg_lambda'] = 5.0

See Also
========

- :doc:`preprocessing` - Data preprocessing
- :doc:`monitoring` - Model monitoring
- :doc:`../guides/model_training` - Model training guide

---

Next: :doc:`monitoring` →

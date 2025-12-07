=============
API Reference
=============

Complete function and class documentation for the VNIT project.

Preprocessing Module
====================

Location: ``src/preprocessing/preprocessing.py``

.. code-block:: python

    from src.preprocessing.preprocessing import (
        load_data,
        basic_eda,
        plot_correlation_heatmap,
        select_important_features,
        check_missing_values,
        preprocess_data,
        main

load_data()
-----------

Load and prepare dataset for preprocessing.

**Signature**::

    load_data(file_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

Parameters
    - ``file_path`` (str): Path to CSV file

Returns
    - Tuple of (train_data, test_data, target)

Raises
    - FileNotFoundError: If file not found
    - ValueError: If required columns missing

**Example**::

    from src.preprocessing.preprocessing import load_data
    train, test, target = load_data('data/train.csv')
    print(f"Loaded: {train.shape[0]} samples, {train.shape[1]} features")

basic_eda()
-----------

Perform exploratory data analysis on dataset.

**Signature**::

    basic_eda(data: pd.DataFrame, target: pd.Series) -> None

Parameters
    - ``data`` (pd.DataFrame): Input features
    - ``target`` (pd.Series): Target variable

Returns
    - None (prints statistics)

**Example**::

    from src.preprocessing.preprocessing import load_data, basic_eda
    train, test, target = load_data('data/train.csv')
    basic_eda(train, target)

plot_correlation_heatmap()
--------------------------

Generate and save correlation heatmap.

**Signature**::

    plot_correlation_heatmap(
        data: pd.DataFrame,
        target: pd.Series,
        figsize: tuple = (16, 12)
    ) -> None

Parameters
    - ``data`` (pd.DataFrame): Input features
    - ``target`` (pd.Series): Target variable
    - ``figsize`` (tuple): Figure dimensions (default: (16, 12))

Returns
    - None (saves figure to disk)

Output
    - Saves: ``correlation_matrix_heatmap.png``

select_important_features()
---------------------------

Select features with correlation above threshold.

**Signature**::

    select_important_features(
        data: pd.DataFrame,
        target: pd.Series,
        threshold: float = 0.5
    ) -> list[str]

Parameters
    - ``data`` (pd.DataFrame): Input features
    - ``target`` (pd.Series): Target variable
    - ``threshold`` (float): Minimum correlation (default: 0.5)

Returns
    - list: Selected feature names

**Example**::

    from src.preprocessing.preprocessing import (
        load_data,
        select_important_features
    train, test, target = load_data('data/train.csv')
    important_cols = select_important_features(train, target, threshold=0.5)
    print(f"Selected {len(important_cols)} features")

check_missing_values()
----------------------

Identify and report missing values in dataset.

**Signature**::

    check_missing_values(data: pd.DataFrame) -> pd.DataFrame

Parameters
    - ``data`` (pd.DataFrame): Input data

Returns
    - pd.DataFrame: Missing value statistics

**Example**::

    from src.preprocessing.preprocessing import (
        load_data,
        check_missing_values
    train, test, target = load_data('data/train.csv')
    missing_report = check_missing_values(train)
    print(missing_report)

preprocess_data()
-----------------

Complete data preprocessing pipeline.

**Signature**::

    preprocess_data(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        target: pd.Series,
        selected_features: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Parameters
    - ``train_data`` (pd.DataFrame): Training data
    - ``test_data`` (pd.DataFrame): Test data
    - ``target`` (pd.Series): Target variable
    - ``selected_features`` (list): Features to include

Returns
    - Tuple of (X_train, X_test, y_train, y_test) as numpy arrays

**Example**::

    from src.preprocessing.preprocessing import (
        load_data,
        select_important_features,
        preprocess_data
    train, test, target = load_data('data/train.csv')
    features = select_important_features(train, target)
    X_train, X_test, y_train, y_test = preprocess_data(
        train, test, target, features
    print(f"X_train shape: {X_train.shape}")

main()
------

Execute full preprocessing pipeline (Prefect orchestrated).

**Signature**::

    main(file_path: str = '../../data/train.csv') -> None

Parameters
    - ``file_path`` (str): Path to data file (default: '../../data/train.csv')

Returns
    - None

**Example**::

    # Run from command line
    cd src/preprocessing
    python preprocessing.py ../../data/train.csv
    # Or programmatically
    from src.preprocessing.preprocessing import main
    main('data/train.csv')

Monitoring Module
=================

Location: ``src/preprocessing/monitoring.py``

.. code-block:: python

    from src.preprocessing.monitoring import (
        create_quality_report,
        create_drift_report,
        validate_data_schema,
        generate_summary_report

create_quality_report()
-----------------------

Generate data quality report (JSON format).

**Signature**::

    create_quality_report(
        data: pd.DataFrame,
        data_name: str = 'dataset'
    ) -> dict

Parameters
    - ``data`` (pd.DataFrame): Data to analyze
    - ``data_name`` (str): Name for report (default: 'dataset')

Returns
    - dict: Quality metrics

**Example**::

    from src.preprocessing.monitoring import create_quality_report
    import pandas as pd
    data = pd.read_csv('data/train.csv')
    report = create_quality_report(data, 'training_data')
    print(f"Missing values: {report['missing_values_count']}")

create_drift_report()
---------------------

Detect data drift between reference and current data.

**Signature**::

    create_drift_report(
        reference_data: pd.DataFrame,
        current_data: pd.DataFrame,
        data_name: str = 'dataset'
    ) -> dict

Parameters
    - ``reference_data`` (pd.DataFrame): Reference dataset
    - ``current_data`` (pd.DataFrame): Current dataset
    - ``data_name`` (str): Name for report

Returns
    - dict: Drift metrics for each feature

**Example**::

    from src.preprocessing.monitoring import create_drift_report
    import pandas as pd
    train = pd.read_csv('data/train.csv').head(1000)
    test = pd.read_csv('data/test.csv')
    drift_report = create_drift_report(train, test, 'housing_data')
    print(f"Drift detected: {drift_report['drift_detected']}")

validate_data_schema()
----------------------

Validate dataset structure and data types.

**Signature**::

    validate_data_schema(
        data: pd.DataFrame,
        expected_schema: dict = None
    ) -> dict

Parameters
    - ``data`` (pd.DataFrame): Data to validate
    - ``expected_schema`` (dict): Expected column/dtype mapping

Returns
    - dict: Validation results

**Example**::

    from src.preprocessing.monitoring import validate_data_schema
    import pandas as pd
    data = pd.read_csv('data/train.csv')
    schema = validate_data_schema(data)
    print(f"Valid: {schema['is_valid']}")

generate_summary_report()
-------------------------

Generate comprehensive statistical summary.

**Signature**::

    generate_summary_report(
        data: pd.DataFrame,
        target: pd.Series = None,
        output_file: str = None
    ) -> str

Parameters
    - ``data`` (pd.DataFrame): Input data
    - ``target`` (pd.Series): Target variable (optional)
    - ``output_file`` (str): File to save report

Returns
    - str: Report text

**Example**::

    from src.preprocessing.monitoring import generate_summary_report
    import pandas as pd
    data = pd.read_csv('data/train.csv')
    target = data['SalePrice']
    report = generate_summary_report(data, target)
    print(report)

Modeling Module
===============

Location: ``src/modeling/modeling.py``

.. code-block:: python

    from src.modeling.modeling import (
        split_data,
        optimize_hyperparameters,
        train_model,
        generate_predictions,
        evaluation,
        get_feature_importance,
        save_model,
        log_to_mlflow,
        generate_performance_report,
        main

split_data()
------------

Split data into training and testing sets.

**Signature**::

    split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

Parameters
    - ``X`` (np.ndarray): Features
    - ``y`` (np.ndarray): Target
    - ``test_size`` (float): Test split ratio (default: 0.2)
    - ``random_state`` (int): Random seed (default: 42)

Returns
    - Tuple of (X_train, X_test, y_train, y_test)

**Example**::

    from src.modeling.modeling import split_data
    import numpy as np
    X = np.random.rand(1000, 50)
    y = np.random.rand(1000)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

optimize_hyperparameters()
--------------------------

Run Optuna hyperparameter optimization.

**Signature**::

    optimize_hyperparameters(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_trials: int = 50
    ) -> dict

Parameters
    - ``X_train`` (np.ndarray): Training features
    - ``y_train`` (np.ndarray): Training target
    - ``X_test`` (np.ndarray): Testing features
    - ``y_test`` (np.ndarray): Testing target
    - ``n_trials`` (int): Number of optimization trials (default: 50)

Returns
    - dict: Best hyperparameters found

Hyperparameter Space

.. list-table::
   :header-rows: 1

   * - Parameter
     - Min
     - Max
     - Type
   * - n_estimators
     - 500
     - 2000
     - int
   * - learning_rate
     - 0.01
     - 0.2
     - float
   * - max_depth
     - 3
     - 15
     - int
   * - min_child_weight
     - 1
     - 10
     - int
   * - gamma
     - 0
     - 5
     - float
   * - subsample
     - 0.5
     - 1.0
     - float
   * - colsample_bytree
     - 0.5
     - 1.0
     - float
   * - reg_alpha
     - 0
     - 10
     - float
   * - reg_lambda
     - 0
     - 10
     - float

train_model()
-------------

Train XGBoost model with given hyperparameters.

**Signature**::

    train_model(
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: dict
    ) -> xgboost.XGBRegressor

Parameters
    - ``X_train`` (np.ndarray): Training features
    - ``y_train`` (np.ndarray): Training target
    - ``params`` (dict): Hyperparameters

Returns
    - xgboost.XGBRegressor: Trained model

**Example**::

    from src.modeling.modeling import train_model
    import numpy as np
    X_train = np.random.rand(900, 50)
    y_train = np.random.rand(900)
    params = {
        'n_estimators': 1000,
        'learning_rate': 0.1,
        'max_depth': 5
    model = train_model(X_train, y_train, params)

generate_predictions()
----------------------

Generate predictions on new data.

**Signature**::

    generate_predictions(
        model: xgboost.XGBRegressor,
        X: np.ndarray
    ) -> np.ndarray

Parameters
    - ``model`` (xgboost.XGBRegressor): Trained model
    - ``X`` (np.ndarray): Features to predict

Returns
    - np.ndarray: Predictions

**Example**::

    from src.modeling.modeling import generate_predictions
    predictions = generate_predictions(model, X_test)
    print(f"Predictions shape: {predictions.shape}")

evaluation()
------------

Evaluate model on test set.

**Signature**::

    evaluation(
        y_test: np.ndarray,
        y_pred: np.ndarray
    ) -> dict

Parameters
    - ``y_test`` (np.ndarray): True values
    - ``y_pred`` (np.ndarray): Predicted values

Returns
    - dict: Metrics (MAE, MSE, RMSE, R²)

Metrics Explained

- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual
- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE (in original units)
- **R² Score**: Proportion of variance explained (0-1, higher is better)

**Example**::

    from src.modeling.modeling import evaluation
    import numpy as np
    y_test = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    metrics = evaluation(y_test, y_pred)
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²: {metrics['r2']:.4f}")

get_feature_importance()
------------------------

Extract feature importance from model.

**Signature**::

    get_feature_importance(
        model: xgboost.XGBRegressor,
        feature_names: list = None,
        top_n: int = 10
    ) -> pd.DataFrame

Parameters
    - ``model`` (xgboost.XGBRegressor): Trained model
    - ``feature_names`` (list): Feature names (optional)
    - ``top_n`` (int): Top N features to return (default: 10)

Returns
    - pd.DataFrame: Feature importance table

**Example**::

    from src.modeling.modeling import get_feature_importance
    importance = get_feature_importance(model, top_n=10)
    print(importance.head())

save_model()
------------

Save trained model to disk.

**Signature**::

    save_model(
        model: xgboost.XGBRegressor,
        filepath: str = '../../models/xgboost_model.joblib'
    ) -> None

Parameters
    - ``model`` (xgboost.XGBRegressor): Model to save
    - ``filepath`` (str): Save location (default: '../../models/xgboost_model.joblib')

Returns
    - None

**Example**::

    from src.modeling.modeling import save_model
    save_model(model, '../../models/my_model.joblib')

log_to_mlflow()
---------------

Log model and metrics to MLflow.

**Signature**::

    log_to_mlflow(
        model: xgboost.XGBRegressor,
        params: dict,
        metrics: dict,
        feature_names: list = None,
        run_name: str = 'XGBoost Regression'
    ) -> None

Parameters
    - ``model`` (xgboost.XGBRegressor): Model to log
    - ``params`` (dict): Hyperparameters
    - ``metrics`` (dict): Evaluation metrics
    - ``feature_names`` (list): Feature names (optional)
    - ``run_name`` (str): MLflow run name

Returns
    - None

**Example**::

    from src.modeling.modeling import log_to_mlflow
    log_to_mlflow(
        model,
        params={'n_estimators': 1000, 'learning_rate': 0.1},
        metrics={'rmse': 28650, 'r2': 0.899},
        feature_names=feature_list,
        run_name='XGBoost v1'

generate_performance_report()
-----------------------------

Generate Evidently performance report.

**Signature**::

    generate_performance_report(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        metrics: dict,
        output_file: str = 'reports/model_performance.json'
    ) -> None

Parameters
    - ``y_test`` (np.ndarray): True values
    - ``y_pred`` (np.ndarray): Predictions
    - ``metrics`` (dict): Evaluation metrics
    - ``output_file`` (str): Report save location

Returns
    - None

main()
------

Execute complete modeling pipeline.

**Signature**::

    main() -> None

Parameters
    - None

Returns
    - None

**Example**::

    # Run from command line
    cd src/modeling
    python modeling.py
    # Or programmatically
    from src.modeling.modeling import main
    main()

Pipeline Stages

1. Data Split
2. Hyperparameter Optimization (Optuna)
3. Model Training (XGBoost)
4. Predictions
5. Evaluation
6. Feature Importance
7. Model Saving
8. MLflow Logging
9. Performance Report

Constants & Configuration
=========================

XGBoost Defaults

.. code-block:: python

    DEFAULT_PARAMS = {
        'objective': 'reg:squarederror',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0,
        'eval_metric': 'rmse'

Optuna Configuration

.. code-block:: python

    N_TRIALS = 50
    TIMEOUT = 600  # seconds

Feature Scaling

- Numeric features: StandardScaler (mean=0, std=1)
- Categorical features: One-hot encoding
- Missing values: Handled in preprocessing

Data Paths

- Training data: ``data/train.csv``
- Model save: ``models/xgboost_model.joblib``
- Reports: ``src/*/reports/``
- MLflow tracking: ``mlruns/``

Common Workflows
================

Workflow 1: End-to-End Training
-------------------------------

::

    # Preprocess data
    cd src/preprocessing
    python preprocessing.py ../../data/train.csv
    # Train model
    cd ../modeling
    python modeling.py
    # View results
    mlflow ui

Workflow 2: Custom Hyperparameters
----------------------------------

Modify ``src/modeling/modeling.py``:

::

    # In optimize_hyperparameters function
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        direction='minimize',
        storage=None,
        load_if_exists=False
    # Modify hyperparameter ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15),

Workflow 3: Model Evaluation
----------------------------

::

    from src.modeling.modeling import (
        evaluation,
        get_feature_importance
    from joblib import load
    model = load('../../models/xgboost_model.joblib')
    metrics = evaluation(y_test, model.predict(X_test))
    importance = get_feature_importance(model, top_n=15)

Workflow 4: Make Predictions
----------------------------

::

    from src.preprocessing.preprocessing import main as preprocess
    from joblib import load
    import numpy as np
    # Preprocess new data
    preprocess('new_data.csv')
    # Load model
    model = load('../../models/xgboost_model.joblib')
    # Predict
    X_new = np.load('X_new.npy')
    predictions = model.predict(X_new)

Error Handling
==============

Common Errors

1. **ModuleNotFoundError**

   ::

       export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

2. **FileNotFoundError**

   ::

       # Ensure data files exist in data/ directory
       ls -la data/

3. **MemoryError**

   ::

       # Reduce dataset size or increase RAM
       # Or use GPU acceleration

4. **Sklearn/Scipy version conflicts**

   ::

       pip install --upgrade scikit-learn scipy xgboost

---

**For more information**, see :doc:`modules/preprocessing` and :doc:`modules/modeling` for detailed function documentation.

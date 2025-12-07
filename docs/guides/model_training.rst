====================
Model Training Guide
====================

This guide covers training the XGBoost model, Optuna tuning, and MLflow tracking.

Quick run
---------

::

    cd src/modeling
    python modeling.py

This runs the full 9-stage pipeline (split, optimize, train, evaluate, save, report).

Customizing hyperparameter search
---------------------------------

Edit ``optimize_hyperparameters`` in ``src/modeling/modeling.py`` to change:

- ``n_trials`` (default 50)
- search ranges for ``n_estimators``, ``learning_rate``, ``max_depth``, and regularization

MLflow
------

The pipeline logs runs to the local ``mlruns/`` directory. To view runs:

::

    mlflow ui
    # then open http://localhost:5000

Production tips
---------------

- Save the final model artifact and its preprocessing objects (scaler, feature list).
- Use the saved model in ``models/xgboost_model.joblib`` for inference.

Evaluation
----------

Evaluation metrics are saved to ``src/modeling/reports/model_performance_*.json`` and include MAE, RMSE and R². Use these to monitor drift over time by comparing to new test sets.

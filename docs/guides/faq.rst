===
FAQ
===

Q: Where are the trained models?
A: In the top-level ``models/`` folder (``models/xgboost_model.joblib``).

Q: How do I view MLflow runs?
A: Run ``mlflow ui`` and open the URL it displays.

Q: How can I speed up hyperparameter tuning?
A: Reduce ``n_trials`` in the Optuna study or sample the dataset.

Q: Where are monitoring reports saved?
A: Check ``src/*/reports/`` for JSON monitoring output.

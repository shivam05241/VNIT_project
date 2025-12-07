========================
Quick Start Guide
========================

Get up and running with the VNIT Housing Price Prediction project in 5 minutes!

Step 1: Installation (1 minute)
===============================

::

    # Clone repository
    git clone https://github.com/shivam05241/VNIT_project.git
    cd VNIT_project
    
    # Create virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    
    # Install dependencies
    pip install -r requirements.txt

Step 2: Prepare Data (30 seconds)
=================================

Download the Kaggle Housing Dataset and place it in the ``data/`` directory:

::

    data/
    └── train.csv        # Your training data

Step 3: Run Preprocessing (1 minute)
====================================

::

    cd src/preprocessing
    python preprocessing.py ../../data/train.csv

Output
------

::

    ============================================================
    PREFECT DATA PREPROCESSING PIPELINE
    ============================================================

    ✓ Loaded data from ../../data/train.csv with shape (1460, 81)
    ✓ Data split: 1168 train, 292 test
    ✓ Selected 18 important features
    ✓ Preprocessing complete - Features: (1460, 41), Target: (1460,)

Step 4: Train Model (2-3 minutes)
=================================

::

    cd ../modeling
    python modeling.py

Output
------

::

    ======================================================================
    MODEL TRAINING PIPELINE WITH PREFECT & EVIDENTLY
    ======================================================================

    STAGE 1: DATA SPLIT
    ✓ Data split: 1168 train, 292 test

    STAGE 2: HYPERPARAMETER OPTIMIZATION
    Starting Optuna optimization with 50 trials...
    [========================================] 100%
    ✓ Best RMSE: 28650.36
    ✓ Best Params: {...}

    STAGE 3: MODEL TRAINING
    ✓ Model training complete!

    STAGE 4: PREDICTIONS
    ✓ Generated 292 predictions

    STAGE 5: MODEL EVALUATION
    ✓ Model Evaluation:
      MAE:  17279.64
      RMSE: 27836.04
      R²:   0.8990

    STAGE 6: FEATURE IMPORTANCE
    ✓ Top 10 Important Features:
       feature         importance
       OverallQual     0.234200
       GarageCars      0.202071
       ...

    STAGE 7-9: Model saved, MLflow logged, Reports generated

Step 5: Make Predictions (30 seconds)
=====================================

::

    cd ../app
    streamlit run streamlit_app.py

Then open ``http://localhost:8501`` in your browser.

What Happens Behind the Scenes
==============================

Preprocessing Pipeline
----------------------

1. Loads 1,460 house records with 81 features
2. Performs EDA and correlation analysis
3. Selects 18 important features (>0.50 correlation)
4. Encodes categorical variables (one-hot encoding)
5. Scales numeric features (StandardScaler)
6. Creates 41 final features

Modeling Pipeline
-----------------

1. Splits data: 1,168 train / 292 test
2. Optimizes hyperparameters with Optuna (50 trials)
3. Trains XGBoost with best parameters
4. Evaluates on test set
5. Logs experiments to MLflow
6. Generates Evidently monitoring reports
7. Saves model for production

Key Files Generated
===================

After running the pipelines:

::

    VNIT_project/
    ├── models/
    │   └── xgboost_model.joblib        # Trained model
    ├── data/
    │   └── X_train.csv                 # Preprocessed features
    ├── src/preprocessing/
    │   └── reports/
    │       ├── data_quality_report_*.json
    │       ├── data_drift_report_*.json
    │       └── Summary Report_*.txt
    ├── src/modeling/
    │   └── reports/
    │       └── model_performance_*.json
    └── mlruns/                         # MLflow experiments

View Results
============

View Experiment History
-----------------------

::

    mlflow ui
    # Open http://localhost:5000

View Data Quality Reports
-------------------------

::

    cat src/preprocessing/reports/Summary\ Report_*.txt

View Model Performance
----------------------

::

    cat src/modeling/reports/model_performance_*.json

Makefile (Optional)
===================

Create a ``Makefile`` for one-command execution:

::

    all: preprocess train

    preprocess:
        cd src/preprocessing && python preprocessing.py ../../data/train.csv

    train:
        cd src/modeling && python modeling.py

    app:
        streamlit run src/app/streamlit_app.py

    monitor:
        cd src/preprocessing && python monitoring.py ../../data/train.csv

    mlflow:
        mlflow ui --port 5000

    clean:
        rm -rf src/*/reports/*.json
        rm -rf models/*

Run with: ``make all``

Common Use Cases
================

Use Case 1: Train Model
-----------------------

::

    cd src/modeling
    python modeling.py

Use Case 2: Monitor Data Quality
--------------------------------

::

    cd src/preprocessing
    python monitoring.py ../../data/train.csv

Use Case 3: Custom Hyperparameters
----------------------------------

Modify ``src/modeling/modeling.py`` optimize_hyperparameters function.

Use Case 4: Make Predictions
----------------------------

::

    streamlit run src/app/streamlit_app.py

Use Case 5: Evaluate on New Data
--------------------------------

::

    python monitoring.py ../../data/test.csv ../../data/train.csv

Performance Benchmarks
======================

.. list-table::
   :header-rows: 1

   * - Stage
     - Time
     - Resources
   * - Preprocessing
     - ~2 seconds
     - <500 MB RAM
   * - Hyperparameter Optimization
     - ~30 seconds
     - 2-4 GB RAM
   * - Model Training
     - ~5 seconds
     - 2-4 GB RAM
   * - Total Pipeline
     - ~40 seconds
     - 2-4 GB RAM

Next Steps
==========

Now that you've run a quick example:

1. **Explore Modules**: Read :doc:`modules/preprocessing` and :doc:`modules/modeling`
2. **Customize**: Modify hyperparameters or features in the code
3. **Deploy**: Use Docker to containerize your application
4. **Monitor**: Set up continuous monitoring with Evidently
5. **Production**: Deploy Streamlit app to cloud platform

Tips & Tricks
=============

1. **Reduce Training Time**: Decrease ``n_trials`` parameter
2. **GPU Acceleration**: Install GPU-enabled XGBoost
3. **Parallel Processing**: Increase ``n_jobs`` parameter
4. **Monitor Progress**: Watch Optuna trials in real-time
5. **Save Experiments**: All runs logged to MLflow automatically

Troubleshooting Quick Fixes
===========================

**Issue**: ModuleNotFoundError

**Fix**::

    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

**Issue**: Port 5000 already in use

**Fix**::

    mlflow ui --port 5001

**Issue**: Out of memory

**Fix**::

    # Reduce dataset or n_trials
    # Use GPU acceleration if available

Getting Help
============

- **Documentation**: Read full docs in :doc:`index`
- **API Reference**: Check :doc:`api_reference`
- **Examples**: See module files with docstrings
- **Issues**: Open GitHub issue

---

Congratulations! You've successfully run the VNIT Housing Price Prediction pipeline! 🎉

**Next**: Explore :doc:`installation` for detailed setup or :doc:`modules/preprocessing` for module documentation.

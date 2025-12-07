===============
Getting Started
===============

Welcome to the VNIT Housing Price Prediction project! This guide will help you get up and running quickly.

Prerequisites
=============

Before you begin, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- Docker (optional, for containerized deployment)

System Requirements
===================

- **OS**: Windows, macOS, or Linux
- **RAM**: 4GB minimum (8GB recommended)
- **Disk Space**: 2GB for data and models
- **GPU**: Optional (for faster training)

What You'll Learn
=================

This project covers:

1. **Data Preprocessing** - Cleaning, feature engineering, and data validation
2. **Exploratory Data Analysis** - Understanding data distributions and relationships
3. **Hyperparameter Optimization** - Using Optuna for efficient parameter search
4. **Model Training** - Training XGBoost with cross-validation
5. **Performance Monitoring** - Using Evidently for data/model drift detection
6. **Workflow Orchestration** - Prefect for reliable pipeline execution
7. **Experiment Tracking** - MLflow for reproducible machine learning
8. **Deployment** - Streamlit for interactive predictions

Project Highlights
==================

- **Modular Design**: Separate preprocessing, modeling, and monitoring modules
- **Production Ready**: Docker support and Streamlit deployment
- **Explainability**: Feature importance analysis and model interpretation
- **Monitoring**: Continuous data quality and model performance tracking
- **Documentation**: Complete API reference and usage examples
- **Best Practices**: Follows ML engineering standards and conventions

Next Steps
==========

1. Read :doc:`installation` for setup instructions
2. Follow :doc:`quick_start` for your first prediction
3. Explore :doc:`modules/preprocessing` to understand data preprocessing
4. Review :doc:`modules/modeling` for model training details
5. Check :doc:`guides/deployment` for production deployment

Key Concepts
============

Pipeline Architecture
---------------------

The project is built as a modular ML pipeline with clear separation of concerns:

::

    Data Loading → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment

Prefect Workflow
----------------

Each stage is implemented as Prefect tasks that can be:

- Monitored and retried automatically
- Executed in parallel where possible
- Logged for audit trails
- Scheduled and triggered

MLflow Experiment Tracking
--------------------------

All experiments are tracked with:

- Model parameters and hyperparameters
- Performance metrics (MAE, MSE, RMSE, R²)
- Model artifacts and metadata
- Git commit information

Evidently Monitoring
--------------------

Production models are monitored for:

- Data drift detection
- Model performance degradation
- Data quality issues
- Distribution shifts

Troubleshooting
===============

**Issue**: Import errors for preprocessing module

**Solution**: Ensure PYTHONPATH includes the src directory::

    export PYTHONPATH="${PYTHONPATH}:/path/to/VNIT_project/src"

**Issue**: Optuna optimization takes too long

**Solution**: Reduce n_trials parameter or increase n_jobs for parallel processing

**Issue**: MLflow UI not accessible

**Solution**: Check if port 5000 is already in use::

    mlflow ui --port 5001

Tips & Best Practices
=====================

1. **Use Virtual Environments**: Always use venv or conda for isolation
2. **Monitor GPU Usage**: Use nvidia-smi to monitor GPU during training
3. **Save Checkpoints**: Models are automatically saved after training
4. **Log Everything**: Use MLflow to track all experiments
5. **Version Your Data**: Keep track of data preprocessing versions
6. **Document Changes**: Update docstrings when modifying code

Resources
=========

- `XGBoost Documentation <https://xgboost.readthedocs.io/>`_
- `Optuna Framework <https://optuna.org/>`_
- `MLflow Documentation <https://mlflow.org/docs/>`_
- `Prefect Documentation <https://docs.prefect.io/>`_
- `Evidently Documentation <https://docs.evidentlyai.com/>`_
- `Scikit-learn Guide <https://scikit-learn.org/>`_

Common Workflows
================

Train a New Model
-----------------

::

    cd src/modeling
    python modeling.py

Monitor Data Quality
--------------------

::

    cd src/preprocessing
    python monitoring.py ../../data/train.csv

Make Predictions
----------------

::

    streamlit run src/app/streamlit_app.py

View Experiment History
-----------------------

::

    mlflow ui

Support
=======

Need help? Check these resources:

- Project README: ``/home/shivam/workspace/VNIT_project/README.md``
- Issue Tracker: GitHub Issues
- Documentation: This Sphinx documentation
- Examples: ``src/`` module files with docstrings

---

Ready to get started? Head to :doc:`installation` →

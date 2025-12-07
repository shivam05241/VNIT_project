================
Monitoring
================

This page describes the data quality and model monitoring features in this project.

Overview
--------

The project includes a lightweight monitoring integration using Evidently (when available) and JSON fallback reports. Monitoring provides:

- Data quality reports (missing values, basic statistics)
- Drift detection between a reference dataset and incoming/current dataset
- Model performance reports (predictions vs. truth, metrics over time)

Files
-----

- ``src/preprocessing/monitoring.py`` — functions to generate data quality and drift reports.
- ``src/modeling/reporting`` (if present) — model performance report helpers.

How to run monitoring
---------------------

1. Generate a data quality report for a dataset:

.. code-block:: bash

    cd src/preprocessing
    python monitoring.py ../../data/train.csv

2. Compare current data to reference data (drift):

.. code-block:: bash

    cd src/preprocessing
    python monitoring.py ../../data/current.csv --reference ../../data/train.csv

3. Generate model performance report (from modeling outputs):

.. code-block:: bash

    cd src/modeling
    python modeling.py --report-only

Notes
-----

- If Evidently is not installed or not compatible, the scripts fall back to producing JSON summary reports in ``src/*/reports/``.
- Reports include timestamps in filenames; check ``src/*/reports/`` for the latest JSON files.

Interpretation
--------------

- data_quality_report_*.json: contains per-column missing counts and basic stats.
- data_drift_report_*.json: contains per-column drift score and a boolean drift flag.
- model_performance_*.json: contains evaluation metrics (MAE, RMSE, R²) and sample histograms.

Adding a monitoring alert
-------------------------

You can add a simple alerting rule by parsing the JSON drift report and sending a notification when "drift_detected" is true for any important feature.

Example (pseudo):

.. code-block:: python

    import json
    r = json.load(open('src/preprocessing/reports/data_drift_report_2025-12-06.json'))
    if any(v['drift_detected'] for v in r['columns'].values()):
        send_alert('Data drift detected')

Related docs
------------

See :doc:`/guides/data_preprocessing` and :doc:`/guides/model_training` for how monitoring fits into the pipeline.

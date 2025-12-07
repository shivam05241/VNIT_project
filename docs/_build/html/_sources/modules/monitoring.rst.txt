.. _module-monitoring:

=================
Monitoring Module
=================

Location: ``src/preprocessing/monitoring.py``

Overview
--------

This module contains helpers to produce data quality and drift reports.

Main functions
~~~~~~~~~~~~~~

- ``create_quality_report(data: pd.DataFrame, data_name: str = 'dataset') -> dict``
  - Generates a JSON-friendly summary of missing values and column statistics.
- ``create_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, data_name: str = 'dataset') -> dict``
  - Compares distributions and returns drift indicators per column.
- ``validate_data_schema(data: pd.DataFrame, expected_schema: dict = None) -> dict``
  - Validates column names and dtypes against an expected schema.

Examples
--------

.. code-block:: python

    from src.preprocessing.monitoring import create_quality_report
    import pandas as pd

    df = pd.read_csv('data/train.csv')
    report = create_quality_report(df, data_name='training_data')
    print(report['missing_values_count'])

Notes
-----

If Evidently is installed the module will attempt to produce Evidently reports; otherwise it will write JSON reports to ``src/preprocessing/reports/``.
=================
Monitoring Module
=================

Location: ``src/preprocessing/monitoring.py``

Overview
--------

This module contains helpers to produce data quality and drift reports.

Main functions
~~~~~~~~~~~~~~

- ``create_quality_report(data: pd.DataFrame, data_name: str = 'dataset') -> dict``
   - Generates a JSON-friendly summary of missing values and column statistics.
- ``create_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame, data_name: str = 'dataset') -> dict``
   - Compares distributions and returns drift indicators per column.
- ``validate_data_schema(data: pd.DataFrame, expected_schema: dict = None) -> dict``
   - Validates column names and dtypes against an expected schema.

Examples
--------

.. code-block:: python

      from src.preprocessing.monitoring import create_quality_report
      import pandas as pd

      df = pd.read_csv('data/train.csv')
      report = create_quality_report(df, data_name='training_data')
      print(report['missing_values_count'])

Notes
-----

If Evidently is installed the module will attempt to produce Evidently reports; otherwise it will write JSON reports to ``src/preprocessing/reports/``.

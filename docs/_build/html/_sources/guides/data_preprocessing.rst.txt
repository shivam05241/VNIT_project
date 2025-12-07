========================
Data Preprocessing Guide
========================

This guide explains the preprocessing pipeline and how to run/extend it.

Quick run
---------

::

    cd src/preprocessing
    python preprocessing.py ../../data/train.csv

This executes the Prefect-orchestrated pipeline and writes reports to ``src/preprocessing/reports/``.

Steps
-----

1. Load data
2. Basic EDA and correlation analysis
3. Select important features
4. Fill missing values and encode categoricals
5. Scale numeric features
6. Output preprocessed ``X_train.csv`` / ``X_test.csv`` used by modeling

Custom runs
-----------

- To run only feature selection or only scaling, import the module and call the function directly:

.. code-block:: python

    from src.preprocessing.preprocessing import select_important_features, preprocess_data
    train, test, target = load_data('data/train.csv')
    cols = select_important_features(train, target, threshold=0.5)
    X_train, X_test, y_train, y_test = preprocess_data(train, test, target, cols)

Troubleshooting
---------------

- If you see missing columns errors, ensure your CSV contains the expected columns (see ``data/data_description.txt``).
- For performance issues, reduce the number of trials or sample the dataset.

Where outputs go
----------------

- Preprocessed features: ``data/X_train.csv`` and ``data/X_test.csv`` (or ``src/preprocessing/reports`` for intermediate artifacts).

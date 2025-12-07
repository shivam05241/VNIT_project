========================
Installation
========================

This guide covers installation of the VNIT Housing Price Prediction project.

Option 1: Local Installation
============================

Step 1: Clone the Repository
----------------------------

::

    git clone https://github.com/shivam05241/VNIT_project.git
    cd VNIT_project

Step 2: Create Virtual Environment
----------------------------------

Using venv::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

Or using conda::

    conda create -n vnit python=3.10
    conda activate vnit

Step 3: Install Dependencies
----------------------------

::

    pip install --upgrade pip
    pip install -r requirements.txt

Step 4: Verify Installation
---------------------------

::

    python -c "import pandas; import xgboost; import optuna; import mlflow; print('✓ All dependencies installed')"

Option 2: Docker Installation
=============================

Step 1: Build Docker Image
--------------------------

::

    docker build -t vnit_project:latest .

Step 2: Run Container
---------------------

::

    docker run -p 8501:8501 vnit_project:latest

Step 3: Access Application
--------------------------

Open browser to ``http://localhost:8501``

Option 3: Docker Compose
========================

Step 1: Run Services
--------------------

::

    docker-compose up --build

Step 2: Access Services
-----------------------

- Streamlit App: ``http://localhost:8501``
- MLflow UI: ``http://localhost:5000``

Dependency Versions
===================

Key dependencies and their versions:

.. list-table::
   :header-rows: 1

   * - Package
     - Version
     - Purpose
   * - numpy
     - >=1.21.0
     - Numerical computing
   * - pandas
     - >=1.3.0
     - Data manipulation
   * - scikit-learn
     - >=1.0.0
     - ML utilities
   * - xgboost
     - >=1.5.0
     - Gradient boosting
   * - optuna
     - >=2.10.0
     - Hyperparameter optimization
   * - mlflow
     - >=1.20.0
     - Experiment tracking
   * - prefect
     - >=2.0.0
     - Workflow orchestration
   * - evidently
     - >=0.1.0
     - ML monitoring
   * - streamlit
     - >=1.0.0
     - Web app framework

System-Specific Installation
============================

For macOS
---------

::

    # Install Xcode command line tools
    xcode-select --install
    
    # Install dependencies with conda (recommended)
    conda create -n vnit python=3.10
    conda activate vnit
    conda install -c conda-forge scikit-learn xgboost
    pip install -r requirements.txt

For Ubuntu/Debian
-----------------

::

    # Install system dependencies
    sudo apt-get install python3-dev python3-pip
    sudo apt-get install build-essential
    
    # Create and activate virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Install Python dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

For Windows
-----------

::

    # Create and activate virtual environment
    python -m venv venv
    venv\Scripts\activate
    
    # Install dependencies
    python -m pip install --upgrade pip
    pip install -r requirements.txt

GPU Support (Optional)
======================

For CUDA GPU acceleration:

Install CUDA Toolkit
--------------------

1. Download from `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-toolkit>`_
2. Install according to OS instructions
3. Verify installation: ``nvidia-smi``

Install GPU-enabled Packages
----------------------------

::

    # For XGBoost with GPU
    pip install xgboost[gpu]
    
    # For RAPIDS (GPU-accelerated data processing)
    pip install -c rapidsai -c conda-forge rapids

Troubleshooting Installation
============================

**Issue**: ``ModuleNotFoundError: No module named 'xgboost'``

**Solution**::

    pip install --upgrade xgboost

**Issue**: ``pip install`` fails with permission error

**Solution**::

    # Use --user flag
    pip install --user -r requirements.txt
    
    # Or activate virtual environment properly
    source venv/bin/activate

**Issue**: ``ImportError: cannot import name 'XGBRegressor'``

**Solution**::

    pip uninstall xgboost -y
    pip install xgboost==1.7.0

**Issue**: CUDA version mismatch

**Solution**::

    # Check CUDA version
    nvcc --version
    
    # Install compatible XGBoost version
    pip install xgboost[gpu]

Verify Installation
===================

Run the verification script:

::

    python -c "
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    import optuna
    import mlflow
    import prefect
    from sklearn.model_selection import train_test_split
    print('✓ pandas:', pd.__version__)
    print('✓ numpy:', np.__version__)
    print('✓ xgboost:', xgb.__version__)
    print('✓ optuna:', optuna.__version__)
    print('✓ mlflow:', mlflow.__version__)
    print('✓ prefect:', prefect.__version__)
    print('✓ All dependencies installed successfully!')
    "

Next Steps
==========

After successful installation:

1. Download the dataset from Kaggle Housing Dataset
2. Place training data in ``data/train.csv``
3. Follow :doc:`quick_start` guide
4. Read module documentation in :doc:`modules/preprocessing`

Advanced Configuration
======================

Set MLflow Backend Store
------------------------

::

    export MLFLOW_BACKEND_STORE_URI="postgresql://user:password@localhost/mlflow"
    export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://mybucket/artifacts"

Configure Optuna Study
----------------------

See :doc:`modules/modeling` for Optuna configuration options.

Enable Logging
--------------

::

    import logging
    logging.basicConfig(level=logging.DEBUG)

---

Ready to start? Head to :doc:`quick_start` →

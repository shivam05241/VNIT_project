# VNIT Project Documentation

## Overview
The Project is a data science project that involves data preprocessing, exploratory data analysis (EDA), and modeling on Housing dataset. The project is structured to facilitate easy understanding and modularity.

## Project Structure
```
VNIT_project/
├── docker-compose.yml
├── Dockerfile
├── README.docker.md
├── README.md
├── requirements.txt
├── data/
│   ├── data_description.txt
│   ├── sample_submission.csv
│   ├── test.csv
│   ├── train.csv
│   ├── X_train.csv
├── models/
│   ├── xgboost_model.joblib
├── src/
│   ├── EDADataPreProcessing.ipynb
│   ├── Modeling.ipynb
│   └── app/
│       └── streamlit_app.py
└── mlruns/
    └── ...
```

## Documentation of Notebooks
### EDADataPreProcessing.ipynb
This notebook is responsible for data preprocessing, including:
- Installing necessary libraries
- Importing packages
- Preprocessing data for analysis

### Modeling.ipynb
This notebook includes:
- Exploratory Data Analysis (EDA)
- Data visualization techniques
- Model training and evaluation

## Requirements
The project requires the following libraries:
- List of libraries from `requirements.txt`

## Usage
Instructions on how to run the project, including local setup, Docker usage, and MLflow.

### Prerequisites
- Docker and docker-compose (optional for containerized runs)
- Python 3.8+ and `pip` (for local runs)
- (Optional) `virtualenv` or `venv` to create an isolated environment

### Run locally (virtual environment)
.. code-block:: bash

    # create and activate virtual environment
    python3 -m venv venv
    source venv/bin/activate

    # install dependencies
    pip install --upgrade pip
    pip install -r requirements.txt

    # run the Streamlit app
    streamlit run src/app/streamlit_app.py

The Streamlit app will be available at `http://localhost:8501` by default.

### Run with Docker
If you prefer to run the project in a containerized environment, use Docker.

.. code-block:: bash

    # build the image (from repository root)
    docker build -t team_19_docker_image:latest .

    # run the container (expose Streamlit port 8501)
    docker run --rm -p 8501:8501 team_19_docker_image:latest

Alternatively, use `docker-compose` if provided in the repository:

.. code-block:: bash

    # start services defined in docker-compose.yml
    docker-compose up --build

Access the app at `http://localhost:8501` after the container starts.

.. code-block:: bash

    # to use already containerized image
    docker load -i team_19_docker_image.tar

### MLflow (view experiments)
If you want to inspect MLflow runs stored in the `mlruns/` directory, run the MLflow UI locally:

.. code-block:: bash

    # from repository root
    mlflow ui --backend-store-uri ./mlruns --port 5000

Then open `http://localhost:5000` to view experiments and run artifacts.

### Common troubleshooting
- If Docker build fails due to permissions, ensure your user can use the Docker socket or run with `sudo`.
- If a dependency fails to install, check `requirements.txt` for platform-specific packages and consult their docs.
- If Streamlit does not appear at `localhost:8501`, confirm the container maps that port and the app started successfully.

### Notes
- The file `models/xgboost_model.joblib` contains a trained model used by the app; the Streamlit app loads it at runtime.
- For production deployments, consider serving the model with a dedicated model server or converting to a lightweight API.

## License
Information about the project's license.

## Authors
- Shivam Awasthi
- Abhishiek Bhadauria

## Acknowledgments
- Any acknowledgments or references used in the project.

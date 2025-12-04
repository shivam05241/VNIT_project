Dockerization notes

This file explains how to build and run the project with Docker and docker-compose.

What the compose setup does
- app: builds the Streamlit app image from the project root and serves the UI on port 8501.
- mlflow: runs a lightweight MLflow server on port 5000. It uses the local ./mlruns folder as the backend store and artifact root.

Quick start (Windows cmd.exe)
1) Build and start both services:

    docker-compose up --build

2) Open the Streamlit UI at:

    http://localhost:8501

   Open the MLflow UI at:

    http://localhost:5000

Notes & advice
- Volumes: docker-compose mounts ./models and ./data into the container as read-only so the app can load the model and sample data without baking them into the image. If you prefer the model inside the image, remove models/ from .dockerignore and copy it in the Dockerfile (not recommended for large files).

- Model path: The Streamlit app uses a relative path from `src/app/streamlit_app.py` to `../../models/xgboost_model.joblib`. The compose volumes map `./models` to `/app/models`, and the app WORKDIR is `/app/src/app`, so the relative path resolves correctly inside the container.

- MLflow persistence: mlruns is mounted to the host to persist experiments across container restarts. If you want a database backend (for production), change the mlflow service to use a dedicated database instead of local files.

- Rebuild after requirements changes: If you change `requirements.txt`, re-run `docker-compose up --build` to pick up new packages.

Troubleshooting
- If the Streamlit image build fails due to missing system libs for some package, add apt-get install lines in the Dockerfile before pip install (keep minimal).
- If MLflow is slow to start the first time, it's because the mlflow package is installed at container start. For faster startup, create a small Dockerfile for the mlflow service that pre-installs mlflow.

Advanced: create a dedicated Dockerfile for the mlflow service to avoid installing on every start. If you'd like, I can add that.

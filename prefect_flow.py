from prefect import flow, task
import subprocess


@task(name="Preprocess data")
def run_preprocessing():
    """
    Runs the data preprocessing script.
    Assumes src/preprocessing/preprocess.py exists
    and creates data/processed/train.csv & test.csv.
    """
    result = subprocess.run(
        ["python", "src/preprocessing/preprocess.py"],
        check=False
    )
    if result.returncode != 0:
        raise RuntimeError("Preprocessing step failed")


@task(name="Train model")
def run_training():
    """
    Runs the model training script.
    Assumes src/training/train.py exists
    and saves models/house_price_model.joblib.
    """
    result = subprocess.run(
        ["python", "src/training/train.py"],
        check=False
    )
    if result.returncode != 0:
        raise RuntimeError("Training step failed")


@flow(name="house-price-mlops-pipeline")
def main_flow():
    """
    Prefect flow that orchestrates the ML pipeline:
    1) preprocessing
    2) model training
    """
    run_preprocessing()
    run_training()


if __name__ == "__main__":
    main_flow()

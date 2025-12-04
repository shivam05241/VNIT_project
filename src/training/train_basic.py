import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

# Paths
RAW_TRAIN_PATH = "data/raw/train.csv"
MODEL_PATH = "models/house_price_model.joblib"

def main():
    # 1. Load data
    df = pd.read_csv(RAW_TRAIN_PATH)

    # 2. Select a few simple features
    features = ["OverallQual", "GrLivArea", "GarageCars",
                "TotalBsmtSF", "FullBath", "YearBuilt"]
    target = "SalePrice"

    # Drop rows with missing values in used columns
    df = df[features + [target]].dropna()

    X = df[features]
    y = df[target]

# 3. Train a simple RandomForest model
    model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
    model.fit(X, y)

# 4. Quick training error (confirm it runs)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    rmse = mse ** 0.5
    print(f"Training RMSE: {rmse:.2f}")

    # 5. Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

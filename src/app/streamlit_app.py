import streamlit as st
import pandas as pd
import joblib
import os

MODEL_PATH = "models/house_price_model.joblib"

st.title("🏠 House Price Prediction – Ames Dataset")

# Check model file
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run: python src/training/train_basic.py")
    st.stop()

# Load model
model = joblib.load(MODEL_PATH)
st.success("Model loaded successfully!")

st.markdown("Enter house details and click **Predict Price**:")

# Input widgets
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea, sq ft)", min_value=200, max_value=6000, value=1500)
garage_cars = st.slider("Garage Capacity (GarageCars)", 0, 5, 2)
total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF, sq ft)", min_value=0, max_value=3000, value=800)
full_bath = st.slider("Number of Full Bathrooms (FullBath)", 0, 4, 2)
year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)

if st.button("Predict Price"):
    data = {
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf],
        "FullBath": [full_bath],
        "YearBuilt": [year_built],
    }
    df = pd.DataFrame(data)
    price = model.predict(df)[0]
    st.success(f"Estimated Sale Price: ${price:,.0f}")

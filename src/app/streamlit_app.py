import streamlit as st
import pandas as pd
import joblib
import os

print(os.getcwd())
base_path = os.path.dirname(os.path.abspath(__file__))
print("base_path:", base_path)
MODEL_PATH = os.path.join(base_path, "../../models/xgboost_model.joblib")
DATA_PATH = os.path.join(base_path, "../../data/X_train.csv")

st.title("🏠 House Price Prediction – Ames Dataset")

# Load model
model = joblib.load(MODEL_PATH)
data_train = pd.read_csv(DATA_PATH)
all_features = model.feature_names_in_
features = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF',
       '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars',
       'GarageArea', 'MSZoning_C (all)', 'MSZoning_FV', 'MSZoning_RH',
       'MSZoning_RL', 'MSZoning_RM', 'Utilities_AllPub',
       'Utilities_NoSeWa', 'BldgType_1Fam', 'BldgType_2fmCon',
       'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE',
       'Heating_Floor', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav',
       'Heating_OthW', 'Heating_Wall', 'KitchenQual_Ex', 'KitchenQual_Fa',
       'KitchenQual_Gd', 'KitchenQual_TA', 'SaleCondition_Abnorml',
       'SaleCondition_AdjLand', 'SaleCondition_Alloca',
       'SaleCondition_Family', 'SaleCondition_Normal',
       'SaleCondition_Partial', 'LandSlope_Gtl', 'LandSlope_Mod',
       'LandSlope_Sev']

st.success("Model loaded successfully!")

st.markdown("Enter house details and click **Predict Price**:")

# Input widgets
overall_qual = st.slider("Overall Quality (1–10)", 1, 10, 5)
gr_liv_area = st.number_input("Above Ground Living Area (GrLivArea, sq ft)", min_value=200, max_value=6000, value=1500)
garage_cars = st.slider("Garage Capacity (GarageCars)", 0, 5, 2)
total_bsmt_sf = st.number_input("Total Basement Area (TotalBsmtSF, sq ft)", min_value=0, max_value=3000, value=800)
full_bath = st.slider("Number of Full Bathrooms (FullBath)", 0, 4, 2)
year_built = st.number_input("Year Built", min_value=1870, max_value=2025, value=2000)
KitchenQual_Ex = st.selectbox("Kitchen Quality - Excellent (KitchenQual_Ex)", [0,1], format_func=lambda x: True if x==1 else False)
KitchenQual_TA = st.selectbox("Kitchen Quality - Typical/Average (KitchenQual_TA)", [0,1], format_func=lambda x: True if x==1 else False)

if st.button("Predict Price"):
    
    data = {
        "OverallQual": [overall_qual],
        "GrLivArea": [gr_liv_area],
        "GarageCars": [garage_cars],
        "TotalBsmtSF": [total_bsmt_sf],
        "FullBath": [full_bath],
        "YearBuilt": [year_built],
        "KitchenQual_TA" : [KitchenQual_TA],
        "KitchenQual_Ex" : [KitchenQual_Ex]
    }

    for features in all_features:
        if features not in data:
            data[features] = [data_train[features].mean()]

    df = pd.DataFrame(data)
    price = model.predict(df[all_features])[0]
    st.success(f"Estimated Sale Price: ${price:,.0f}")

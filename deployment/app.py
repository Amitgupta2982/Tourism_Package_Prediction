import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -------------------------------------------------------
# DOWNLOAD AND LOAD THE TOURISM MODEL FROM HUGGINGFACE
# -------------------------------------------------------
model_path = hf_hub_download(
    repo_id="Amitgupta2982/Tourism-Package-Model",
    filename="tourism_xgb_best_model_v1.joblib"
)
model = joblib.load(model_path)

# -------------------------------------------------------
# STREAMLIT USER INTERFACE
# -------------------------------------------------------
st.title("Tourism Package Purchase Prediction App")

st.write("""
This interactive application predicts whether a customer is likely to purchase a tourism package.

Please enter the customer's demographic, income, and interaction details below to generate the prediction.
""")

# -------------------------------------------------------
# USER INPUT FIELDS
# -------------------------------------------------------
Age = st.number_input("Age", min_value=18, max_value=80, value=35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (Minutes)", min_value=0, max_value=50, value=10)
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=1)
PreferredPropertyStar = st.selectbox("Preferred Hotel Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
Passport = st.selectbox("Passport Available?", [0, 1])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Own Car?", [0, 1])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
Designation = st.selectbox(
    "Customer Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)
MonthlyIncome = st.number_input("Monthly Income (in local currency)", min_value=3000, max_value=300000, value=50000)

# -------------------------------------------------------
# ASSEMBLE INPUT FOR THE MODEL
# -------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# -------------------------------------------------------
# PREDICTION BUTTON
# -------------------------------------------------------
if st.button("Predict Tourism Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "LIKELY to Purchase" if prediction == 1 else "NOT Likely to Purchase"

    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import shap
import matplotlib.pyplot as plt

# Load model and scaler
model = pickle.load(open("voting_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

xgb_model = model.named_estimators_["xgb"]
lr_model = model.named_estimators_["lr"]

features = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk', 'Sex', 'Age', 'Education', 'Income'
]

xgb_explainer = shap.Explainer(xgb_model)

st.set_page_config(page_title="Diabetes Risk Predictor", layout="wide")
st.markdown("<h1 style='text-align: center;'>🩺 Diabetes Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("### Please answer the following questions:")

with st.form("diabetes_form"):
    user_input = {}
    for i in range(0, len(features), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(features):
                feature = features[i + j]

                if feature == "BMI":
                    user_input[feature] = cols[j].slider("Body Mass Index (BMI)", 10, 60, 25)

                elif feature == "MentHlth":
                    user_input[feature] = cols[j].slider("Days mental health not good (last 30 days)", 0, 30, 0)

                elif feature == "PhysHlth":
                    user_input[feature] = cols[j].slider("Days physical health not good (last 30 days)", 0, 30, 0)

                elif feature == "GenHlth":
                    user_input[feature] = cols[j].selectbox("Rate your general health", [1, 2, 3, 4, 5],
                        format_func=lambda x: ["Excellent", "Very Good", "Good", "Fair", "Poor"][x - 1])

                elif feature == "Age":
                    user_input[feature] = cols[j].selectbox(
                        "Age group", list(range(1, 14)),
                        format_func=lambda x: {
                            1: "18–24", 2: "25–29", 3: "30–34", 4: "35–39", 5: "40–44",
                            6: "45–49", 7: "50–54", 8: "55–59", 9: "60–64", 10: "65–69",
                            11: "70–74", 12: "75–79", 13: "80+"
                        }[x]
                    )

                elif feature == "Education":
                    user_input[feature] = cols[j].selectbox(
                        "Education level", list(range(1, 7)),
                        format_func=lambda x: [
                            "Never attended school", "Grades 1–8", "Grades 9–11",
                            "High school graduate", "Some college", "College graduate"
                        ][x - 1]
                    )

                elif feature == "Income":
                    user_input[feature] = cols[j].selectbox(
                        "Income range", list(range(1, 9)),
                        format_func=lambda x: [
                            "< $10k", "$10k–15k", "$15k–20k", "$20k–25k",
                            "$25k–35k", "$35k–50k", "$50k–75k", "> $75k"
                        ][x - 1]
                    )

                elif feature == "Sex":
                    user_input[feature] = 1 if cols[j].radio("Biological Sex", ["Female", "Male"]) == "Female" else 0

                else:
                    questions = {
                        "HighBP": "Do you have high blood pressure?",
                        "HighChol": "Do you have high cholesterol?",
                        "CholCheck": "Had cholesterol checked in last 5 years?",
                        "Smoker": "Smoked at least 100 cigarettes in life?",
                        "Stroke": "Have you had a stroke?",
                        "HeartDiseaseorAttack": "Heart disease or heart attack history?",
                        "PhysActivity": "Physical activity in last 30 days?",
                        "Fruits": "Eat fruits at least once daily?",
                        "Veggies": "Eat vegetables daily?",
                        "HvyAlcoholConsump": "Heavy alcohol consumption?",
                        "AnyHealthcare": "Have any health insurance?",
                        "NoDocbcCost": "Couldn't see doctor due to cost?",
                        "DiffWalk": "Difficulty walking/climbing stairs?"
                    }
                    user_input[feature] = cols[j].radio(questions.get(feature, feature), ["No", "Yes"]) == "Yes"

    submit = st.form_submit_button("🔍 Predict")

if submit:
    input_data = pd.DataFrame([{k: int(v) if isinstance(v, bool) else v for k, v in user_input.items()}])
    input_data = input_data[features]  # Ensure correct column order
    input_data = input_data.astype(float)  # Ensure numeric consistency

    st.write("🔎 Model Input Data:", input_data)

    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes\n🧮 Confidence Score: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Diabetes\n🧮 Confidence Score: {probability:.2f}")

    st.markdown("### 🔍 SHAP Explainability")

    with st.expander("🌲 XGBoost SHAP"):
        xgb_shap_values = xgb_explainer(scaled_input)
        fig1, ax1 = plt.subplots()
        shap.plots.waterfall(xgb_shap_values[0], max_display=10, show=False)
        st.pyplot(fig1)

    with st.expander("📈 Logistic Regression SHAP"):
        lr_explainer = shap.LinearExplainer(lr_model, masker=scaled_input, feature_perturbation="interventional")
        lr_shap_values = lr_explainer(scaled_input)
        fig2, ax2 = plt.subplots()
        shap.plots.waterfall(lr_shap_values[0], max_display=10, show=False)
        st.pyplot(fig2)

    input_data["Prediction"] = prediction
    input_data["Confidence"] = round(probability, 2)
    input_data["Timestamp"] = datetime.datetime.now()

    with open("logs.csv", "a") as f:
        input_data.to_csv(f, header=f.tell() == 0, index=False)
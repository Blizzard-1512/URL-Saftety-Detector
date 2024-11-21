import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load models
try:
    vtc = pickle.load(open("vtc.pkl", "rb"))
    dtc = pickle.load(open("dtc.pkl", "rb"))
    rf = pickle.load(open("rf.pkl", "rb"))
    bcf = pickle.load(open("bcf.pkl", "rb"))
    xgb = pickle.load(open("xgb.pkl", "rb"))
    abc = pickle.load(open("abc.pkl", "rb"))
    svm = pickle.load(open("svm.pkl", "rb"))
    lr = pickle.load(open("lr.pkl", "rb"))
    model_path = "final_model.h5"
    model = load_model(model_path)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# Load test data
try:
    test_data = pickle.load(open("test_data.pkl", "rb"))
    X_test = test_data["X"]
    y_test = test_data["y"]
except FileNotFoundError:
    st.warning("Test data file 'test_data.pkl' not found. Accuracy calculations will not be performed.")
    X_test, y_test = None, None

# App title
st.title("URL Safety Prediction using Machine Learning & Deep Learning")

# Steps for using the application
st.subheader("Steps to use the URL Legitimacy Detector")
st.markdown("""
1. **Copy any URL** you want to test for safety.  
2. **Select a model** from the sidebar.  
   - Recommended: **Random Forest Classifier** for accuracy.  
3. Enter the URL and click **Predict** to see the results and insights.
""")

# Input URL
st.header("Enter a URL")
url_input = st.text_input("Input the URL:")

# Feature extraction
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0
    features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
    features["Shortining_Service"] = 1 if "bit.ly" in url or "t.co" in url else 0
    features["having_At_Symbol"] = 1 if "@" in url else 0
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
    features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
    features["having_Sub_Domain"] = -1 if parsed_url.netloc.count(".") > 2 else (0 if parsed_url.netloc.count(".") == 2 else 1)
    features["SSLfinal_State"] = 1 if url.startswith("https") else -1
    features["Google_Index"] = 1 if "google.com" in url else 0
    features["Submitting_to_email"] = 1 if "mailto:" in url else 0
    features["Abnormal_URL"] = -1 if len(parsed_url.netloc.split(".")) > 3 else 1
    return features

if url_input:
    st.write("Extracting features from the URL...")
    extracted_features = extract_features(url_input)
    feature_values = np.array([[extracted_features[key] for key in extracted_features]])
else:
    extracted_features = None

# Sidebar model selection
st.sidebar.header("Select Models for Prediction")
models = {
    "Voting Classifier": vtc,
    "Decision Trees": dtc,
    "Random Forests (Better for Generalization)": rf,
    "Bagging Classifier (Better for Consistency)": bcf,
    "XGBoost Classifier": xgb,
    "AdaBoost Classifier": abc,
    "Support Vector Classifier": svm,
    "Neural Networks": model
}
selected_models = []
for model_name in models:
    if st.sidebar.checkbox(model_name):
        selected_models.append((model_name, models[model_name]))

# Prediction button
if st.button("Predict") and extracted_features:
    predictions = {}
    for model_name, model in selected_models:
        try:
            if hasattr(model, "predict_proba"):
                prediction_probs = model.predict_proba(feature_values)[:, 1]
                prediction_class = (prediction_probs > 0.5).astype(int)
            else:
                prediction_class = model.predict(feature_values)
            if model_name == "Neural Networks":
                prediction_probs = model.predict(feature_values)
                prediction_class = (prediction_probs > 0.5).astype(int)
            predictions[model_name] = "Safe" if prediction_class[0] == 1 else "Malicious"
        except NotFittedError:
            st.error(f"Model {model_name} is not properly fitted.")
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")

    # Display predictions
    prediction_df = pd.DataFrame([{"Model": name, "Prediction": result} for name, result in predictions.items()])
    st.dataframe(prediction_df.style.apply(
        lambda row: [f"background-color: {'green' if row.Prediction == 'Safe' else 'red'}"] * len(row), axis=1
    ))

    # Feature importance plot
    feature_importances = pd.Series(extracted_features).sort_values(ascending=False)[:5]
    plt.style.use("seaborn-darkgrid")
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Top 5 Contributing Features", fontsize=16, fontweight="bold")
    plt.xlabel("Feature", fontsize=14)
    plt.ylabel("Importance Value", fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)

    # Display "Go to URL" button
    if "Safe" in prediction_df["Prediction"].values:
        st.markdown(
            f"""
            <style>
                .go-to-url {{
                    background-color: #4CAF50;
                    border: none;
                    color: white;
                    padding: 10px 20px;
                    text-align: center;
                    text-decoration: none;
                    display: inline-block;
                    font-size: 16px;
                    margin: 4px 2px;
                    cursor: pointer;
                    border-radius: 8px;
                }}
                .go-to-url:hover {{
                    background-color: #45a049;
                }}
            </style>
            <a class="go-to-url" href="{url_input}" target="_blank">Go to URL</a>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Malicious URL detected. Do not visit this link.")
else:
    if not selected_models:
        st.error("Please select at least one model for prediction.")

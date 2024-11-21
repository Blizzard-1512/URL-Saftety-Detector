import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
from urllib.parse import urlparse
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
    model = load_model("final_model.h5")
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}")
    st.stop()

# App title
st.title("URL Safety Prediction using Machine Learning & Deep Learning")

# Sidebar Instructions
st.sidebar.header("Steps to Use")
st.sidebar.markdown("""
1. **Enter any URL** you wish to check for safety.  
2. **Select a model** to use for prediction from the sidebar.  
3. Click **Predict** to see the result and a feature analysis.  
""")

# Input URL
st.header("Enter a URL")
url_input = st.text_input("Paste the URL below:")

# Feature Extraction
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0
    features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
    features["Shortening_Service"] = 1 if "bit.ly" in url or "t.co" in url else 0
    features["having_At_Symbol"] = 1 if "@" in url else 0
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
    features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
    features["having_Sub_Domain"] = -1 if parsed_url.netloc.count(".") > 2 else (0 if parsed_url.netloc.count(".") == 2 else 1)
    features["SSLfinal_State"] = 1 if url.startswith("https") else -1
    features["Google_Index"] = 1 if "google.com" in url else 0
    features["Submitting_to_email"] = 1 if "mailto:" in url else 0
    features["Abnormal_URL"] = -1 if len(parsed_url.netloc.split(".")) > 3 else 1
    return features

# Sidebar Model Selection
st.sidebar.header("Select Models")
models = {
    "Voting Classifier": vtc,
    "Decision Trees": dtc,
    "Random Forests": rf,
    "Bagging Classifier": bcf,
    "XGBoost Classifier": xgb,
    "AdaBoost Classifier": abc,
    "Support Vector Classifier": svm,
    "Neural Networks": model
}
selected_models = []
for model_name in models:
    if st.sidebar.checkbox(model_name):
        selected_models.append((model_name, models[model_name]))

# Predict Button
if st.button("Predict"):
    if not url_input:
        st.error("Please enter a URL before predicting.")
    elif not selected_models:
        st.error("Please select at least one model for prediction.")
    else:
        # Extract features
        extracted_features = extract_features(url_input)
        feature_values = np.array([[extracted_features[key] for key in extracted_features]])

        # Perform predictions
        predictions = {}
        for model_name, model in selected_models:
            try:
                if hasattr(model, "predict"):
                    prediction = model.predict(feature_values)
                    prediction_class = "Safe" if prediction[0] == 1 else "Malicious"
                else:
                    prediction_probs = model.predict(feature_values)
                    prediction_class = "Safe" if prediction_probs[0] > 0.5 else "Malicious"
                predictions[model_name] = prediction_class
            except NotFittedError:
                st.error(f"Model {model_name} is not properly fitted.")
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")

        # Display predictions
        prediction_df = pd.DataFrame(
            [{"Model": name, "Prediction": result} for name, result in predictions.items()]
        )
        st.dataframe(
            prediction_df.style.apply(
                lambda x: ["background-color: green" if v == "Safe" else "background-color: red" for v in x["Prediction"]],
                axis=1,
            )
        )

        # Feature Importance Visualization
        st.subheader("Feature Contribution")
        plt.style.use("seaborn-darkgrid")
        plt.figure(figsize=(10, 6))
        pd.Series(extracted_features).sort_values(ascending=False).plot(
            kind="bar", color="skyblue", edgecolor="black"
        )
        plt.title("Feature Contributions", fontsize=16, fontweight="bold")
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Importance", fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        st.pyplot(plt)

        # Go to URL Button
        if "Safe" in predictions.values():
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
                unsafe_allow_html=True,
            )
        else:
            st.warning("The URL is flagged as Malicious. Avoid visiting this link.")

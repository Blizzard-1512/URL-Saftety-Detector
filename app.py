import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from urllib.parse import urlparse
import socket
import requests
import json

from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model

# Set page configuration to wide mode and default theme
st.set_page_config(page_title="URL Safety Prediction using Machine Learning & Deep Learning", 
                   page_icon=":shield:", 
                   layout="wide", 
                   initial_sidebar_state="expanded")

# Load all models
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

# Attempt to load test data
try:
    test_data = pickle.load(open("test_data.pkl", "rb"))
    X_test = test_data["X"]
    y_test = test_data["y"]
except FileNotFoundError:
    st.warning(
        "Test data file 'test_data.pkl' not found. Accuracy calculations will not be performed."
    )
    X_test, y_test = None, None

# Custom CSS for dark theme and styling
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
.stDataFrame {
    color: #000000;
}
.stTextInput > div > div > input {
    color: #FFFFFF;
    background-color: #262730;
}
h1, h2, h3, h4 {
    color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# Define the app title
st.title("URL Safety Prediction using Machine Learning & Deep Learning")

# Insert steps for using the application
st.subheader("Steps to use the URL Legitimacy Detector")
st.markdown(""" 
1. **Copy any URL** of your choice which you want to test and find out whether it is safe or malicious.  
2. **Select the model** you want to predict the result with from the sidebar on the left.  
   - We recommend using the **Random Forest Classifier** for high accuracy.  
   - Use the **Bagging Classifier** for consistent predictions.  
3. After entering your URL and selecting the model, **click on the "Predict" button** and watch the magic happen.  
   - Our model extracts the actual features from your URL and processes them to test for its legitimacy.  
""")

# URL Input
st.header("Enter a URL")
url_input = st.text_input("Input the URL:")

# Feature extraction function
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)

    # Extract features (same as previous implementation)
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0
    features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
    features["Shortining_Service"] = 1 if "bit.ly" in url or "t.co" in url else 0
    features["having_At_Symbol"] = 1 if "@" in url else 0
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
    features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
    features["having_Sub_Domain"] = -1 if parsed_url.netloc.count(".") > 2 else (
        0 if parsed_url.netloc.count(".") == 2 else 1)
    features["SSLfinal_State"] = 1 if url.startswith("https") else -1
    features["Domain_registeration_length"] = 0  # Placeholder
    features["Favicon"] = 1  # Placeholder
    features["port"] = 1  # Placeholder
    features["HTTPS_token"] = 1 if "https-" in parsed_url.netloc else 0
    features["Request_URL"] = 1  # Placeholder
    features["URL_of_Anchor"] = 0  # Placeholder
    features["Links_in_tags"] = 0  # Placeholder
    features["SFH"] = 0  # Placeholder
    features["Submitting_to_email"] = 1 if "mailto:" in url else 0
    features["Abnormal_URL"] = -1 if len(parsed_url.netloc.split(".")) > 3 else 1
    features["Redirect"] = 1 if "→" in url else -1
    features["on_mouseover"] = 0  # Placeholder
    features["RightClick"] = 0  # Placeholder
    features["popUpWidnow"] = 0  # Placeholder
    features["Iframe"] = 0  # Placeholder
    features["age_of_domain"] = 0  # Placeholder
    features["DNSRecord"] = 0  # Placeholder
    features["web_traffic"] = 0  # Placeholder
    features["Page_Rank"] = 0  # Placeholder
    features["Google_Index"] = 1 if "google.com" in url else 0
    features["Links_pointing_to_page"] = 0  # Placeholder
    features["Statistical_report"] = 0  # Placeholder

    return features

# Extract features when URL is entered
if url_input:
    st.write("Extracting features from the URL...")
    extracted_features = extract_features(url_input)
    feature_values = np.array([[extracted_features[key] for key in extracted_features]])

    # Get top 5 contributing features
    top_features = pd.Series(extracted_features).sort_values(ascending=False)[:5]
    
    st.write("Extracted feature values:")
    for key, value in extracted_features.items():
        st.write(f"{key}: {value}")
else:
    extracted_features = None
    top_features = None

# Sidebar with model selection
st.sidebar.header("Select Models for Prediction")

# Add a separation line between the button and model options
st.sidebar.markdown("<hr>", unsafe_allow_html=True)

# Models dictionary
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

# Preloaded sample malicious URLs
malicious_url_samples = [
    "http://secure-data-access.com",
    "http://account-verification-update.net",
    "http://confirm-details-protect.io",
    "http://secure-login-prompt.org",
    "http://system-update-required.xyz",
    "http://service-check-notification.biz",
    "http://new-login-verification.me",
    "http://confirm-request-processing.cc"
]

# Sidebar: Button to generate a malicious URL
generate_malicious_button = st.sidebar.button("Generate Malicious URL")

# Variable to store generated URL
generated_url = None

# Generate a malicious URL when clicked
if generate_malicious_button:
    generated_url = np.random.choice(malicious_url_samples)
    st.sidebar.write(f"Generated URL: {generated_url}")

    # Use this URL for feature extraction
    extracted_features = extract_features(generated_url)
    feature_values = np.array([[extracted_features[key] for key in extracted_features]])

    # Show the "Copy URL" button after the URL is generated
    if generated_url:
        copy_button = st.sidebar.button("Copy URL to Clipboard", key="copy_url_button")

        if copy_button:
            # JavaScript to copy the URL to the clipboard
            st.sidebar.markdown(f"""
            <script>
            navigator.clipboard.writeText("{generated_url}")
            alert("URL copied to clipboard!");
            </script>
            

# Prediction button and "Go to URL" button
if st.button("Predict") and extracted_features:
    predictions = {}
    if selected_models:
        for model_name, model in selected_models:
            try:
                if hasattr(model, "predict_proba"):  # For models that predict probabilities
                    prediction_probs = model.predict_proba(feature_values)[:, 1]
                    prediction_class = (prediction_probs >= 0.5).astype(int)
                    accuracy = accuracy_score(y_test, prediction_class) * 100 if X_test is not None else None
                else:
                    prediction_class = model.predict(feature_values)
                    accuracy = accuracy_score(y_test, prediction_class) * 100 if X_test is not None else None

                predictions[model_name] = {
                    "Prediction": "Safe" if prediction_class[0] == 1 else "Malicious",
                    "Accuracy": f"{accuracy:.2f}%" if accuracy is not None else "N/A"
                }
            except NotFittedError:
                st.error(f"The model {model_name} is not properly fitted. Please check the model.")
            except Exception as e:
                st.error(f"An error occurred with {model_name}: {e}")

        # Display predictions
        if predictions:
            prediction_df = pd.DataFrame([{
                "Model": name, "Prediction": details["Prediction"], "Accuracy": details["Accuracy"]
            } for name, details in predictions.items()])

            # Highlight safe and malicious predictions
            def highlight_predictions(row):
                color = 'green' if row["Prediction"] == "Safe" else 'red'
                return [f'background-color: {color}; color: white;'] * len(row)

            st.write("Prediction Results:")
            st.dataframe(prediction_df.style.apply(highlight_predictions, axis=1))

            # Display Top Contributing Features
            st.subheader("Top 5 Contributing Features")
            
            # Create a beautiful display of top features
            feature_container = st.container()
            with feature_container:
                cols = st.columns(5)
                for i, (feature, value) in enumerate(top_features.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            background-color: #262730;
                            border-radius: 10px;
                            padding: 10px;
                            text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                            color: #FFFFFF;
                        ">
                        <h4 style="color: #FFFFFF;">{feature}</h4>
                        <p style="color: #CCCCCC; font-size: 14px;">Value: {value}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # Add some vertical space
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

            # Display "Go to URL" button if Safe
            if "Safe" in prediction_df["Prediction"].values:
                # Use custom dark theme button styling
                st.markdown(
                    f'<a href="{url_input}" target="_blank" style="text-decoration: none;">'
                    f'<button style="background-color: #4CAF50; color: white; padding: 10px 20px; '
                    f'border: none; border-radius: 4px; cursor: pointer; font-weight: bold;">Go to URL</button></a>', 
                    unsafe_allow_html=True
                )
            else:
                st.warning("Malicious URL detected. It is recommended not to visit this link.")
    else:
        st.error("No models selected. Please choose at least one model to make predictions.")

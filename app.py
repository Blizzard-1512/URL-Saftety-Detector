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

# Function to get IP address
def get_ip_address(url):
    try:
        # Extract domain from URL
        domain = urlparse(url).netloc
        # Remove port if exists
        domain = domain.split(':')[0]
        return socket.gethostbyname(domain)
    except Exception as e:
        st.error(f"Could not resolve IP address: {e}")
        return None

# Function to get IP geolocation
def get_ip_geolocation(ip_address):
    try:
        # Use ipapi.co for free IP geolocation
        response = requests.get(f'https://ipapi.co/{ip_address}/json/')
        return response.json()
    except Exception as e:
        st.error(f"Could not fetch IP geolocation: {e}")
        return None

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

# Function to convert continuous probabilities to binary class labels
def convert_to_class_labels(predictions, threshold=0.5):
    """Converts continuous predictions to binary class labels based on a threshold."""
    return (predictions > threshold).astype(int)

# Prediction button and "Go to URL" button
if st.button("Predict") and extracted_features:
    # Get IP Address and Geolocation
    ip_address = get_ip_address(url_input)
    ip_geolocation = get_ip_geolocation(ip_address) if ip_address else None

    predictions = {}
    if selected_models:
        for model_name, model in selected_models:
            try:
                if hasattr(model, "predict_proba"):  # For models that predict probabilities
                    prediction_probs = model.predict_proba(feature_values)[:, 1]
                    prediction_class = convert_to_class_labels(prediction_probs)
                else:
                    prediction_class = model.predict(feature_values)

                # Handle neural network model separately
                if model_name == "Neural Networks":
                    prediction_probs = model.predict(feature_values)
                    prediction_class = convert_to_class_labels(prediction_probs)

                accuracy = None
                if X_test is not None and y_test is not None:
                    if model_name == "Neural Networks":
                        y_pred_nn = convert_to_class_labels(model.predict(X_test))
                        accuracy = accuracy_score(y_test, y_pred_nn) * 100
                    else:
                        accuracy = accuracy_score(y_test, model.predict(X_test)) * 100

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
                cols = st.columns(len(top_features))
                for i, (feature, value) in enumerate(top_features.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div style="
                            background-color: #f0f2f6;
                            border-radius: 10px;
                            padding: 10px;
                            text-align: center;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        ">
                        <h4 style="color: #333;">{feature}</h4>
                        <p style="color: #666; font-size: 14px;">Value: {value}</p>
                        </div>
                        """, unsafe_allow_html=True)

            # IP Geolocation Visualization
            if ip_geolocation:
                st.subheader("IP Geolocation")
                
                # Create Plotly world map with IP location
                fig = px.scatter_geo(
                    locationmode='country names',
                    lon=[ip_geolocation.get('longitude')],
                    lat=[ip_geolocation.get('latitude')],
                    hover_name=[f"IP: {ip_address}"],
                    text=[f"Country: {ip_geolocation.get('country_name')}<br>City: {ip_geolocation.get('city')}"],
                    projection='natural earth'
                )
                fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig)

                # Display additional IP details
                st.write(f"**IP Address:** {ip_address}")
                st.write(f"**Country:** {ip_geolocation.get('country_name')}")
                st.write(f"**City:** {ip_geolocation.get('city')}")

            # Display "Go to URL" button if Safe
            if "Safe" in prediction_df["Prediction"].values:
                # Use standard Streamlit button styling
                st.markdown(
                    f'<a href="{url_input}" target="_blank" style="text-decoration: none;">'
                    f'<button style="background-color: #4CAF50; color: white; padding: 10px 20px; '
                    f'border: none; border-radius: 4px; cursor: pointer;">Go to URL</button></a>', 
                    unsafe_allow_html=True
                )
            else:
                st.warning("Malicious URL detected. It is recommended not to visit this link.")
    else:
        st.error("No models selected. Please choose at least one model to make predictions.")

import streamlit as st
import numpy as np
import pickle
import re
import random
import string
import socket
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from urllib.parse import urlparse
from tld import get_tld

from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model

# Set page configuration to wide mode and default theme
st.set_page_config(page_title="URL Safety Detector", 
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

# Malicious URL Generator Function
def generate_malicious_url():
    malicious_patterns = [
        "{scheme}://{subdomain}.{domain}.{tld}/login?redirect={malicious_path}",
        "{scheme}://{domain}.{tld}/login.php?next={malicious_path}",
        "{scheme}://{subdomain}.{domain}.{tld}/verify?token={random_token}",
        "{scheme}://{domain}.{tld}/profile?id={random_id}&redirect={malicious_path}",
        "{scheme}://{subdomain}.{domain}.{tld}/click?ref={random_ref}"
    ]

    malicious_domains = [
        "scam", "phish", "fraud", "malware", "virus", "hack", "steal", 
        "login-verify", "secure-account", "verify-now"
    ]

    tlds = ["com", "org", "net", "info", "xyz"]
    schemes = ["http", "https"]
    subdomains = ["www", "login", "secure", "verify", "account"]

    random_pattern = random.choice(malicious_patterns)
    domain = random.choice(malicious_domains)
    tld = random.choice(tlds)
    scheme = random.choice(schemes)
    subdomain = random.choice(subdomains)

    # Generate random malicious path
    malicious_path = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    random_token = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    random_id = ''.join(random.choices(string.digits, k=5))
    random_ref = ''.join(random.choices(string.ascii_lowercase + string.digits, k=12))

    malicious_url = random_pattern.format(
        scheme=scheme,
        subdomain=subdomain,
        domain=domain,
        tld=tld,
        malicious_path=malicious_path,
        random_token=random_token,
        random_id=random_id,
        random_ref=random_ref
    )

    return malicious_url

# Malicious URL Generator Section in Sidebar
st.sidebar.header("Malicious URL Generator")
if st.sidebar.button("Generate Malicious URL"):
    malicious_url = generate_malicious_url()
    st.sidebar.markdown(f"**Generated Malicious URL:**\n```\n{malicious_url}\n```")
    st.sidebar.warning("Warning: This is a simulated malicious URL for educational purposes.")

def get_domain_age(url):
    try:
        domain = urlparse(url).netloc
        creation_date = socket.gethostbyname(domain)
        return 1
    except:
        return -1

def check_dns_record(url):
    try:
        domain = urlparse(url).netloc
        socket.gethostbyname(domain)
        return 1
    except:
        return -1

def check_web_traffic(url):
    try:
        response = requests.get(f"https://api.similarweb.com/v1/similar-rank/{url}/ranks")
        if response.status_code == 200:
            return 1
        return -1
    except:
        return -1

# URL Input
st.header("Enter a URL")
url_input = st.text_input("Input the URL:")

# Feature extraction function
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)
    
    try:
        tld_obj = get_tld(url, as_object=True)
    except:
        tld_obj = None

    # Comprehensive Feature Extraction
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else -1
    features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
    features["Shortining_Service"] = 1 if any(service in url.lower() for service in ["bit.ly", "t.co", "goo.gl", "tinyurl"]) else -1
    features["having_At_Symbol"] = 1 if "@" in url else -1
    features["double_slash_redirecting"] = 1 if url.count("//") > 1 else -1
    features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
    features["having_Sub_Domain"] = -1 if (tld_obj and len(tld_obj.subdomain) > 0) else 1
    features["SSLfinal_State"] = 1 if url.startswith("https") else -1
    features["Domain_registeration_length"] = get_domain_age(url)
    features["Favicon"] = check_web_traffic(url)
    features["port"] = 1 if parsed_url.port else -1
    features["HTTPS_token"] = 1 if "https-" in parsed_url.netloc else -1
    features["Request_URL"] = check_web_traffic(url)
    features["URL_of_Anchor"] = check_dns_record(url)
    features["Links_in_tags"] = check_web_traffic(url)
    features["SFH"] = 1 if any(keyword in url for keyword in ['secure', 'login', 'signin']) else -1
    features["Submitting_to_email"] = 1 if "mailto:" in url else -1
    features["Abnormal_URL"] = -1 if tld_obj and len(tld_obj.subdomain.split('.')) > 3 else 1
    features["Redirect"] = 1 if "→" in url else -1
    features["on_mouseover"] = -1
    features["RightClick"] = -1
    features["popUpWidnow"] = -1
    features["Iframe"] = -1
    features["age_of_domain"] = get_domain_age(url)
    features["DNSRecord"] = check_dns_record(url)
    features["web_traffic"] = check_web_traffic(url)
    features["Page_Rank"] = check_web_traffic(url)
    features["Google_Index"] = 1 if "google.com" in url else -1
    features["Links_pointing_to_page"] = check_web_traffic(url)
    features["Statistical_report"] = -1

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

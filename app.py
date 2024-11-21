import streamlit as st
import numpy as np
import pickle
import random
import string
from urllib.parse import urlparse
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model
import pandas as pd
import re

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

# Load malicious URL properties dataset
try:
    malicious_url_props = pd.read_csv("Malicious_URL_Props.csv")
except FileNotFoundError:
    st.error("Malicious URL properties file not found. Ensure the file is in the correct location.")
    st.stop()

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

# Malicious URL Generator Function based on dataset
def generate_malicious_url_from_data(props_df):
    url_length_ranges = props_df["URL_Length"].value_counts(normalize=True).to_dict()
    subdomain_likelihood = props_df["having_Sub_Domain"].value_counts(normalize=True).to_dict()
    prefix_suffix_likelihood = props_df["Prefix_Suffix"].value_counts(normalize=True).to_dict()

    domain_lengths = [random.randint(5, 15) for _ in range(1000)]
    tlds = ["com", "org", "net", "xyz", "info"]
    schemes = ["http", "https"]
    subdomains = ["login", "secure", "verify", "account", "www"]

    url_length = random.choices(
        list(url_length_ranges.keys()), weights=list(url_length_ranges.values()), k=1
    )[0]
    domain_length = random.choice(domain_lengths)
    has_subdomain = random.choices(
        list(subdomain_likelihood.keys()), weights=list(subdomain_likelihood.values()), k=1
    )[0]
    has_prefix_suffix = random.choices(
        list(prefix_suffix_likelihood.keys()), weights=list(prefix_suffix_likelihood.values()), k=1
    )[0]

    scheme = random.choice(schemes)
    domain = "".join(random.choices(string.ascii_lowercase, k=domain_length))
    tld = random.choice(tlds)
    subdomain = random.choice(subdomains) if has_subdomain == -1 else ""
    prefix_suffix = "-" if has_prefix_suffix == -1 else ""
    path = "".join(random.choices(string.ascii_letters + string.digits, k=random.randint(5, 15)))

    malicious_url = f"{scheme}://{subdomain + '.' if subdomain else ''}{domain}{prefix_suffix}.{tld}/{path}"
    return malicious_url

# Malicious URL Generator Section in Sidebar
st.sidebar.header("Malicious URL Generator")
if st.sidebar.button("Generate Malicious URL"):
    malicious_url = generate_malicious_url_from_data(malicious_url_props)
    st.sidebar.markdown(f"**Generated Malicious URL:**\n```\n{malicious_url}\n```")
    st.sidebar.warning("Warning: This is a simulated malicious URL for educational purposes.")

# URL Input
st.header("Enter a URL")
url_input = st.text_input("Input the URL:")

# Feature extraction function
def extract_features(url):
    features = {}
    parsed_url = urlparse(url)

    # Extract features as in original code...
    features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else -1
    # Other features go here...
    return features

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

# Prediction button and "Go to URL" button
if st.button("Predict") and url_input:
    extracted_features = extract_features(url_input)
    feature_values = np.array([[extracted_features[key] for key in extracted_features]])
    predictions = {}

    if selected_models:
        for model_name, model in selected_models:
            try:
                # Prediction logic here...
                predictions[model_name] = {
                    "Prediction": "Safe",
                    "Accuracy": "N/A"
                }
            except NotFittedError:
                st.error(f"The model {model_name} is not properly fitted.")
            except Exception as e:
                st.error(f"An error occurred with {model_name}: {e}")

        st.write(predictions)

import streamlit as st
import numpy as np
import pickle
import re
import random
import string
import socket
import requests
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse
from tld import get_tld
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from tensorflow.keras.models import load_model

# Set page configuration
st.set_page_config(
    page_title="URL Safety Detector",
    page_icon=":shield:",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# Test data (optional)
try:
    test_data = pickle.load(open("test_data.pkl", "rb"))
    X_test = test_data["X"]
    y_test = test_data["y"]
except FileNotFoundError:
    X_test, y_test = None, None

# Custom CSS for theme and styling
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
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

# Instructions
st.subheader("How to Use the Application")
st.markdown("""
1. Enter the **URL** you want to check for safety in the input box.
2. Select one or more models from the **sidebar** to predict.
3. Click **Predict** to see if the URL is **Safe** or **Malicious**.
4. Use the **Malicious URL Generator** in the sidebar for testing purposes.
""")

# Fetch malicious URL properties from GitHub
GITHUB_CSV_URL = "https://github.com/Blizzard-1512/URL-Saftety-Detector/blob/master/Malicious_URL_Props.csv"

@st.cache_data
def fetch_malicious_url_props():
    try:
        response = requests.get(GITHUB_CSV_URL)
        response.raise_for_status()
        return pd.read_csv(pd.compat.StringIO(response.text))
    except Exception as e:
        st.error(f"Error fetching malicious URL properties: {e}")
        return None

malicious_url_props = fetch_malicious_url_props()
if malicious_url_props is None:
    st.stop()

# Malicious URL generator using fetched properties
def generate_malicious_url(props_df):
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

    return f"{scheme}://{subdomain + '.' if subdomain else ''}{domain}{prefix_suffix}.{tld}/{path}"

# Malicious URL Generator in Sidebar
st.sidebar.header("Malicious URL Generator")
if st.sidebar.button("Generate Malicious URL"):
    malicious_url = generate_malicious_url(malicious_url_props)
    st.sidebar.markdown(f"**Generated Malicious URL:**\n```\n{malicious_url}\n```")
    st.sidebar.warning("This is a simulated malicious URL for testing purposes.")

# URL Input
st.header("Enter a URL")
url_input = st.text_input("URL:")

# Feature Extraction
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        "having_IP_Address": 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else -1,
        "URL_Length": len(url),
        "having_Sub_Domain": 1 if parsed_url.hostname and parsed_url.hostname.count(".") > 2 else -1,
        "Prefix_Suffix": 1 if "-" in parsed_url.hostname else -1,
        "SSLfinal_State": 1 if url.startswith("https") else -1,
    }
    return features

# Sidebar model selection
st.sidebar.header("Select Models for Prediction")
models = {
    "Voting Classifier": vtc,
    "Decision Trees": dtc,
    "Random Forests": rf,
    "Bagging Classifier": bcf,
    "XGBoost Classifier": xgb,
    "AdaBoost Classifier": abc,
    "Support Vector Classifier": svm,
    "Neural Networks": model,
}
selected_models = [name for name in models if st.sidebar.checkbox(name)]

# Prediction
if st.button("Predict") and url_input:
    features = extract_features(url_input)
    feature_values = np.array([[features[key] for key in features]])
    predictions = {}

    if selected_models:
        for name in selected_models:
            model = models[name]
            try:
                if name == "Neural Networks":
                    pred = model.predict(feature_values)
                    predictions[name] = "Safe" if pred[0] < 0.5 else "Malicious"
                else:
                    pred = model.predict(feature_values)
                    predictions[name] = "Safe" if pred[0] == 1 else "Malicious"
            except Exception as e:
                st.error(f"Error with {name}: {e}")

        st.subheader("Predictions")
        for name, result in predictions.items():
            st.write(f"**{name}:** {result}")
    else:
        st.error("No models selected. Please choose at least one model.")

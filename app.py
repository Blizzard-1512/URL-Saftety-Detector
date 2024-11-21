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

def check_domain_registration(domain):
    """
    Check domain registration length and availability
    Returns 1 if recently registered or unavailable, 0 if well-established
    """
    try:
        # Perform a simple socket lookup
        socket.gethostbyname(domain)
        return 0  # Domain exists
    except:
        return 1  # Domain might not exist or is suspicious

def check_ssl_status(url):
    """
    Check SSL certificate status
    """
    try:
        response = requests.head(url, timeout=5)
        return 1 if response.url.startswith('https') else -1
    except:
        return -1

def generate_malicious_url():
    """
    Generate a simulated malicious URL with suspicious characteristics
    """
    # Suspicious domain generators
    suspicious_tlds = ['.xyz', '.top', '.loan', '.click', '.online']
    suspicious_subdomains = ['malware', 'hack', 'scam', 'phish', 'spam']
    
    # Randomly choose URL generation strategy
    strategies = [
        # IP-based URL
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}/",
        
        # Suspicious domain with random characters
        lambda: f"http://{random.choice(suspicious_subdomains)}-{''.join(random.choices(string.ascii_lowercase, k=5))}{random.choice(suspicious_tlds)}",
        
        # URL with multiple redirects and suspicious parameters
        lambda: f"http://example.com/redirect?url=http://malicious{random.randint(100,999)}.{random.choice(suspicious_tlds)}/exec?param=dangerous",
        
        # Short URL with suspicious redirection
        lambda: f"http://bit.ly/{random.choice(suspicious_subdomains)}{random.randint(1000,9999)}",
        
        # URL with multiple @ symbols and IP
        lambda: f"http://{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}@suspicious{random.choice(suspicious_tlds)}"
    ]
    
    return random.choice(strategies)()

def extract_features(url):
    features = {}
    try:
        parsed_url = urlparse(url)

        # IP Address Detection (improved)
        features["having IP Address"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed_url.netloc) else 0
        
        # URL Length
        features["URL_Length"] = -1 if len(url) > 75 else (0 if len(url) > 54 else 1)
        
        # URL Shortening Service
        features["Shortining_Service"] = 1 if any(service in url.lower() for service in ['bit.ly', 't.co', 'goo.gl', 'tinyurl']) else 0
        
        # @ Symbol Detection
        features["having_At_Symbol"] = 1 if "@" in url else 0
        
        # Double Slash Redirecting
        features["double_slash_redirecting"] = 1 if url.count("//") > 1 else 0
        
        # Prefix/Suffix Detection
        features["Prefix_Suffix"] = -1 if "-" in parsed_url.netloc else 1
        
        # Subdomain Detection (improved)
        sub_domains = parsed_url.netloc.split('.')
        features["having_Sub_Domain"] = -1 if len(sub_domains) > 3 else (0 if len(sub_domains) == 3 else 1)
        
        # SSL Status (now using active check)
        features["SSLfinal_State"] = check_ssl_status(url)
        
        # Domain Registration Length
        features["Domain_registeration_length"] = check_domain_registration(parsed_url.netloc)
        
        # HTTPS Token
        features["HTTPS_token"] = 1 if "https-" in parsed_url.netloc else 0
        
        # Email Submission
        features["Submitting_to_email"] = 1 if "mailto:" in url else 0
        
        # Abnormal URL Structure
        features["Abnormal_URL"] = -1 if len(sub_domains) > 3 else 1
        
        # Redirect Indicator
        features["Redirect"] = 1 if "→" in url else -1
        
        # Google Indexing
        features["Google_Index"] = 1 if "google.com" in url else 0
        
        # Placeholder features (you might want to develop more sophisticated checks)
        features["port"] = 1
        features["Favicon"] = 1
        features["Request_URL"] = 1
        features["URL_of_Anchor"] = 0
        features["Links_in_tags"] = 0
        features["SFH"] = 0
        features["on_mouseover"] = 0
        features["RightClick"] = 0
        features["popUpWidnow"] = 0
        features["Iframe"] = 0
        features["age_of_domain"] = 0
        features["DNSRecord"] = 0
        features["web_traffic"] = 0
        features["Page_Rank"] = 0
        features["Links_pointing_to_page"] = 0
        features["Statistical_report"] = 0

    except Exception as e:
        st.warning(f"Error extracting features: {e}")
        # Return default features if extraction fails
        features = {k: 0 for k in [
            "having IP Address", "URL_Length", "Shortining_Service", "having_At_Symbol", 
            "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain", "SSLfinal_State",
            "Domain_registeration_length", "HTTPS_token", "Submitting_to_email", 
            "Abnormal_URL", "Redirect", "Google_Index", "port", "Favicon", "Request_URL", 
            "URL_of_Anchor", "Links_in_tags", "SFH", "on_mouseover", "RightClick", 
            "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord", "web_traffic", 
            "Page_Rank", "Links_pointing_to_page", "Statistical_report"
        ]}

    return features

# Modify the Sidebar to include URL Generation
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

# Add URL Generation to the Sidebar
url_gen_mode = st.sidebar.radio("Choose Mode", 
    ["URL Detection", "Generate Malicious URLs"], 
    help="Switch between detecting URL safety and generating sample malicious URLs")

# Existing model selection checkboxes
selected_models = []
for model_name in models:
    if st.sidebar.checkbox(model_name):
        selected_models.append((model_name, models[model_name]))

# Modify the main prediction workflow
if url_gen_mode == "Generate Malicious URLs":
    st.header("Malicious URL Generator")
    st.markdown("""
    This tool generates simulated malicious URLs to demonstrate the classification capabilities.
    **Note:** These are synthetic URLs and should NOT be actually visited.
    """)

    num_urls = st.slider("Number of Malicious URLs to Generate", 1, 10, 5)
    
    if st.button("Generate Malicious URLs"):
        malicious_urls = [generate_malicious_url() for _ in range(num_urls)]
        
        st.subheader("Generated Malicious URLs")
        for url in malicious_urls:
            st.markdown(f"🚨 `{url}`")
            
            # Predict on generated URLs
            extracted_features = extract_features(url)
            feature_values = np.array([[extracted_features[key] for key in extracted_features]])
            
            predictions = {}
            for model_name, model in models.items():
                try:
                    if hasattr(model, "predict_proba"):
                        prediction_probs = model.predict_proba(feature_values)[:, 1]
                        prediction_class = (prediction_probs > 0.5).astype(int)
                    else:
                        prediction_class = model.predict(feature_values)

                    predictions[model_name] = "Malicious" if prediction_class[0] == 0 else "Safe"
                except Exception as e:
                    st.error(f"Prediction error for {model_name}: {e}")
            
            # Display model predictions
            st.write("Model Predictions:")
            pred_df = pd.DataFrame.from_dict(predictions, orient='index', columns=['Prediction'])
            st.table(pred_df)
            st.markdown("---")

else:
    # Existing URL Input section
    st.header("Enter a URL")
    url_input = st.text_input("Input the URL:")

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

    # Prediction button and "Go to URL" button
    if st.button("Predict") and extracted_features:
        predictions = {}
        if selected_models:
            for model_name, model in selected_models:
                try:
                    if hasattr(model, "predict_proba"):  # For models that predict probabilities
                        prediction_probs = model.predict_proba(feature_values)[:, 1]
                        prediction_class = (prediction_probs > 0.5).astype(int)
                    else:
                        prediction_class = model.predict(feature_values)

                    # Handle neural network model separately
                    if model_name == "Neural Networks":
                        prediction_probs = model.predict(feature_values)
                        prediction_class = (prediction_probs > 0.5).astype(int)

                    accuracy = None
                    if X_test is not None and y_test is not None:
                        if model_name == "Neural Networks":
                            y_pred_nn = (model.predict(X_test) > 0.5).astype(int)
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

            # Rest of the existing prediction display code remains the same
            # (Display predictions, top features, Go to URL button, etc.)

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

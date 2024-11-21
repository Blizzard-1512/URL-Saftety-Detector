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

# Set page configuration FIRST - wide mode and dark theme
st.set_page_config(
    page_title="URL Safety Predictor", 
    page_icon="🔒", 
    layout="wide",  # Default to wide mode
    initial_sidebar_state="expanded"
)

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

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .stDataFrame {
        background-color: #1E2130;
    }
    .stContainer {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .stMarkdown {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Define the app title
st.title("🔒 URL Safety Prediction using Machine Learning & Deep Learning")

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

# [Rest of the code remains the same as the previous implementation until the IP Geolocation Visualization]

# IP Geolocation Visualization
if ip_geolocation:
    st.subheader("IP Geolocation")
    
    # Updated Geolocation Visualization with fixes
    try:
        # Ensure we have valid longitude and latitude
        longitude = ip_geolocation.get('longitude', 0)
        latitude = ip_geolocation.get('latitude', 0)
        
        if longitude and latitude:
            # Create Plotly world map with IP location
            fig = px.scatter_geo(
                locationmode='ISO-3',
                lon=[longitude],
                lat=[latitude],
                hover_name=[f"IP: {ip_address}"],
                color_discrete_sequence=['red'],
                size=[10],  # Making point more visible
                projection='natural earth'
            )
            
            # Customize layout for dark theme
            fig.update_layout(
                height=400, 
                margin={"r":0,"t":0,"l":0,"b":0},
                geo=dict(
                    showland=True,
                    landcolor='rgb(30, 33, 48)',  # Dark background
                    showcountries=True,
                    countrycolor='rgb(100, 100, 100)',  # Border color
                    showocean=True,
                    oceancolor='rgb(14, 17, 23)',  # Dark ocean
                ),
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent
                plot_bgcolor='rgba(0,0,0,0)'    # Transparent
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Could not retrieve valid geolocation coordinates.")
    
    except Exception as e:
        st.error(f"Error in geolocation visualization: {e}")

    # Display additional IP details with improved styling
    ip_details_col1, ip_details_col2 = st.columns(2)
    
    with ip_details_col1:
        st.markdown(f"""
        <div style="background-color: #1E2130; padding: 15px; border-radius: 10px; color: white;">
        <h4>IP Details</h4>
        <p><strong>IP Address:</strong> {ip_address}</p>
        <p><strong>Country:</strong> {ip_geolocation.get('country_name', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with ip_details_col2:
        st.markdown(f"""
        <div style="background-color: #1E2130; padding: 15px; border-radius: 10px; color: white;">
        <h4>Location Information</h4>
        <p><strong>City:</strong> {ip_geolocation.get('city', 'N/A')}</p>
        <p><strong>Region:</strong> {ip_geolocation.get('region', 'N/A')}</p>
        <p><strong>Timezone:</strong> {ip_geolocation.get('timezone', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

# [Rest of the code remains the same as the previous implementation]

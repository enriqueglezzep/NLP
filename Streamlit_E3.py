import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained LSTM model
model = load_model('sentiment_analysis_model.h5')

# Function to preprocess user input
def preprocess_input(text):
    # Tokenization, padding, etc. (similar to what was done during model training)
    # Your preprocessing steps here...
    return processed_text  # Return processed text

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_input(text)
    sequence = np.array([processed_text])
    prediction = model.predict(sequence)[0][0]
    return prediction

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon=":smiley:",
    layout="wide"
)

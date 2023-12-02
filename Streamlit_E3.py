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
    
# Streamlit app
st.title('Twitter Sentiment Analysis')

comment = st.text_input('Enter your comment:')
if st.button('Submit'):
    if comment:
        sentiment = predict_sentiment(comment)
        if sentiment >= 0.5:
            st.write('Sentiment: Positive')
        else:
            st.write('Sentiment: Negative')
    else:
        st.write('Please enter a comment.')

st.markdown(
    """
    <style>
    .big-font {
        font-size: 24px !important;
        color: #0066cc;
    }
    .positive {
        color: #008000;
    }
    .negative {
        color: #ff0000;
    }
    .center {
        text-align: center;
    }
    .logo {
        max-width: 150px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header with logo
st.markdown('<img src="Sentlogo.png" class="logo">', unsafe_allow_html=True)
st.title('Twitter Sentiment Analysis')
st.subheader('Analyze the sentiment of your Twitter comments.')

comment = st.text_input('Enter your comment:')
if st.button('Submit'):
    if comment:
        # Your sentiment analysis logic here
        sentiment = 0.7  # Replace with your actual sentiment prediction

        if sentiment >= 0.5:
            st.markdown('<p class="big-font positive center">Sentiment: Positive</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="big-font negative center">Sentiment: Negative</p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="big-font center">Please enter a comment.</p>', unsafe_allow_html=True)
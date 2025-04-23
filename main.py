# Step 1: Import Libraries and Load the Model
import numpy as np
import json
import h5py
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import model_from_json

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import SimpleRNN, Embedding, Dense, InputLayer

def load_fixed_model(h5_path):
    import h5py, json

    with h5py.File(h5_path, 'r') as f:
        config = f.attrs.get('model_config')
    
    # Parse the JSON
    config_json = json.loads(config)

    # Remove 'time_major' if present in SimpleRNN layers
    for layer in config_json['config']['layers']:
        if layer['class_name'] == 'SimpleRNN' and 'time_major' in layer['config']:
            del layer['config']['time_major']

    # Deserialize using known Keras classes
    model = model_from_json(
        json.dumps(config_json),
        custom_objects={
            'Sequential': Sequential,
            'InputLayer': InputLayer,
            'Embedding': Embedding,
            'SimpleRNN': SimpleRNN,
            'Dense': Dense
        }
    )
    model.load_weights(h5_path)
    return model



model = load_fixed_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Step 3: Streamlit App
import streamlit as st

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]:.4f}')
else:
    st.write('Please enter a movie review.')

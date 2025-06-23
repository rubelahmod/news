import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, Dropout, GlobalMaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.utils import register_keras_serializable

# Custom LSTM to avoid loading error
@register_keras_serializable()
class CustomLSTM(tf.keras.layers.LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        kwargs.pop('implementation', None)
        super().__init__(*args, **kwargs)

# Load model within custom object scope
with tf.keras.utils.custom_object_scope({
    'CustomLSTM': CustomLSTM,
    'LSTM': CustomLSTM,
    'Bidirectional': Bidirectional,
    'Embedding': Embedding,
    'Dense': Dense,
    'Dropout': Dropout,
    'GlobalMaxPooling1D': GlobalMaxPooling1D
}):
    model = load_model("model.h5")

# Set up tokenizer (IMPORTANT: You should load your original tokenizer instead if you saved it)
vocab_size = 10000
maxlen = 177
tok = Tokenizer(num_words=vocab_size)
# You must fit the tokenizer with the original training texts for real use

# Class labels
labels = ['World News', 'Sports News', 'Business News', 'Science-Technology News']

# Streamlit UI
st.set_page_config(page_title="News Category Predictor", layout="centered")
st.title("📰 News Headline Category Classifier")
st.write("Enter a news headline below to classify it into one of the four categories:")

user_input = st.text_area("Your News Headline", height=100)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid headline.")
    else:
        # Tokenize input
        seq = tok.texts_to_sequences([user_input])
        padded_seq = pad_sequences(seq, maxlen=maxlen)

        # Prediction
        prediction = model.predict(padded_seq)
        predicted_class = labels[np.argmax(prediction)]

        # Display result
        st.success(f"📢 Predicted Category: **{predicted_class}**")

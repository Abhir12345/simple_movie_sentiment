import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequences

model = tensorflow.keras.models.load_model('simple_rnn.h5')
st.title("Moview Sentiment Prediction.")

review = st.text_input("Enter the string.", 500)
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

max_len = 500

# Helper functions
def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word,2)+3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review],maxlen = 500)
  return padded_review

def predict_sentiment(review):
  # corrected the typo, using 'review' instead of 'reivew'
  review = preprocess_text(review) 
  prediction = model.predict(review)[0][0]
  if prediction > 0.5:
    sentiment = 'Positive'
  else:
    sentiment = 'Negative'
  return sentiment, prediction

st.write("The sentiment predicted is ", predict_sentiment(review)[0])

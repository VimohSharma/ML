import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
    
word_index=imdb.get_word_index()
reverse_word_index = {value:key for key, value in word_index.items()}

model=load_model('simple_rnn_imdb_model.h5')

## Step 2 - Helper Fxnx
# Fxn to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3,'?')for i in encoded_review])

# Fxn to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

### Prediction fxn step 3

def prediction_sentiment(review):
    prepreocessed_review=preprocess_text(review)
    prediction=model.predict(prepreocessed_review)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment,prediction[0][0]


## Streamlt app
def run():
    import streamlit as st

    st.title("IMDB Movie Review Sentiment Analysis")
    st.write("Enter a movie review to classify it as positive as negative")

    user_input=st.text_area("Movie_Review")

    if(st.button('Classify')):
        prepocessed_input=preprocess_text(user_input)


        prediction= model.predict(prepocessed_input)
        sentiments=''
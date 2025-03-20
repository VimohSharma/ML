import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model=load_model('next_word_LSTM.h5')

with open('tokenizerLSTM.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

## Fxn to predict word

def predcit_next_word(model,tokenizer,text,max_sq_len):

    tokn_list=tokenizer.texts_to_sequences([text])[0]
    if len(tokn_list)>=max_sq_len:
        tokn_list=tokn_list[-(max_sq_len-1):]
    tokn_list=pad_sequences([tokn_list],maxlen=max_sq_len-1,padding='pre')
    predcited=model.predict(tokn_list,verbose=0)
    predcited_word_index=np.argmax(predcited,axis=1)
    for word,index in tokenizer.word_index.items():
        if index == predcited_word_index:
            return word
    return None

def run():
    st.title("Next word prediction with LSTM and Early Stopping")

    input_text=st.text_input("Enter the sequence of words","To be or not to")
    if st.button("Predict the next word"):
        max_sq_len=model.input_shape[1] + 1
        next_word = predcit_next_word(model,tokenizer,input_text,max_sq_len)
        st.write(f'Next word can be: {next_word}')
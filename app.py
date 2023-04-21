#1. Preprocess
#2. Vectorize
#3. Predict
#4. Display
import streamlit as st
import pickle as pickle

import nltk
from nltk.corpus import stopwords
import string 
from nltk.stem.porter import PorterStemmer

# Load the saved models
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model= pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_area("Enter the message ")


# pre process using transform text
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()# convert sentence to lower case
    text = nltk.word_tokenize(text) # tokenize sentence to words- provide a list of tokenized words 
    y = []
    for i in text:
        if i.isalnum(): #Remove special character and allow only alpha numerica words
            y.append(i)
    text = y.copy()
    y.clear()
    
    for i in text: #remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y.copy()
    y.clear()
    for i in text: #Stemming of words
        y.append(ps.stem(i))
        
    return " ".join(y)


if st.button('Predict'):
    #1. Preprocess
    transformed_sms = transform_text(input_sms)
    #2. vectorizer
    vector_input = tfidf.transform([transformed_sms])
    #3. predict
    result = model.predict(vector_input)[0]
    #4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


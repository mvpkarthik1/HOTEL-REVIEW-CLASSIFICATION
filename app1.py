import streamlit as st
from sklearn import *
import nltk
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

nltk.download('stopwords')
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
pickle_in = open("clf_RFC.pkl","rb")
classification = pickle.load(pickle_in)

#---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='HOTEL REVIEW CLASSIFICATION',
    layout='wide')

#---------------------------------#   
st.header("""
HOTEL REVIEW CLASSIFICATION

Enter some text in the input box and check the sentiment analysis

""")

#---------------------------------#
st.markdown('**ENTER THE REVIEW**')
user_input = st.text_input("Review","enter text here")


user_input = re.sub(r'\[[0-9]*\]', ' ',user_input)
user_input = re.sub(r'\s+',' ',user_input)
user_input = user_input.lower()
user_input = re.sub(r'\d',' ',user_input)
user_input = re.sub(r'\s+',' ',user_input)


#text = "Nick likes to play football, however he is not too fond of tennis."
text_tokens = word_tokenize(user_input)

tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]

#print(tokens_without_sw)

#stop = stopwords.words('english')
#user_input =user_input.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
filtered_sentence = (" ").join(tokens_without_sw)
#print(filtered_sentence)
text = ' '.join(filtered_sentence)
review = [text]

vectorizer = TfidfVectorizer()


#print(matrix.shape)
if st.button('Validate Sentiment'):
    matrix = vectorizer.transform(review)
    print(classification.predict(matrix))
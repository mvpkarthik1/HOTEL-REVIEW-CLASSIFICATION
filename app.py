import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
import re
from rake_nltk import Rake

pickle_in = open('vectorizer.pkl', 'rb') 
vectorizer = pickle.load(pickle_in)

#---------------------------------#

pickle_in = open("clf_RFC.pkl","rb")
classification = pickle.load(pickle_in)

#---------------------------------#

def cleantext(text):
    text = re.sub(r"â€™", "", text) # Remove Mentions
    text = re.sub(r"#", "", text) # Remove Hashtags Symbol
    text = re.sub(r"\w*\d\w*", "", text) # Remove numbers
    text = re.sub(r"https?:\/\/\S+", "", text) # Remove The Hyper Link
    text = re.sub(r"______________", "", text) # Remove _____
    
    
    return text


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
text = cleantext(user_input)
text = ' '.join(text)
review  = vectorizer.transform([text])


if st.button('Validate Sentiment'):
    matrix = vectorizer.fit_transform(review).toarray()
    if classification.predict(review) == -1:
            st.write("Input review has Negative Sentiment.:sad:")
    elif classification.predict(review) == 1:
            st.write("Input review has Positive Sentiment.:smile:")
    else:
            st.write(" Input review has Neutral Sentiment.😐")

if st.button("Keywords"):
    st.header("Keywords")    
    r=Rake(language='english')
    r.extract_keywords_from_text(user_input)
    # Get the important phrases
    phrases = r.get_ranked_phrases()
    # Display the important phrases
    st.write("These are the **keywords** causing the above sentiment:")
    for i, p in enumerate(phrases):
        st.write(i+1, p)
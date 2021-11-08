# -*- coding: utf-8 -*-
"""
Created on Sun Aug 29 15:00:25 2021

@author: Aatif Zeya
"""

import pandas as pd
import streamlit as st 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfTransformer
import spacy
import nltk

st.title('Model Deployment: RandomForestClassifier ')



st.sidebar.header('Write the Review below')
reviews = st.sidebar.text_area("Text ","")
def user_input_features():
    
    data = {"reviews": reviews
    }
    features = pd.DataFrame(data,index=[0])
    return features 


daf = user_input_features()
st.subheader('User Input details')
st.write(daf)
    
    

df = pd.read_csv("C:/Users/NAMRATA/Downloads/Deployment2.csv")

X = df['reviews']
y = df['sentiment_type']

   # Extract Feature With TfidfVectorizer
tv=TfidfVectorizer()
X=tv.fit_transform(df['reviews'])   

from imblearn.over_sampling import SMOTE
sm = SMOTE(k_neighbors=2)
X_sm, Y_sm = sm.fit_resample(X,y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, Y_sm, test_size=0.3, random_state=100)

RFC = RandomForestClassifier()
RFC.fit(X_train,y_train)
RFC.score(X_test,y_test)
newadd = reviews
new = [newadd]
vect = tv.transform(new).toarray()
prediction = RFC.predict(vect)
   
st.subheader('Predicted Result')
st.write("Positive Sentiment" if prediction>0.01 else "Negative Sentiment")

from nltk.corpus import stopwords
df_idf = df
stop_words = list(stopwords.words('english'))
docs=df_idf['reviews'].tolist()


cv = CountVectorizer(max_df=0.85,max_features=400000)
word_count_vector=cv.fit_transform(docs)

tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(word_count_vector)



df_test = daf
docs_test=df_test['reviews'].tolist()

# get test docs into a list
feature_names=cv.get_feature_names()
doc=docs_test[0]
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    
        #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
        
        
        #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results




























# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st 
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfTransformer
import spacy

st.title('Model Deployment: LinearSVC ')

st.sidebar.header('Write the Review below')
sentence = st.sidebar.text_area("Text ","")
def user_input_features():
    
    
    data = {"sentence": sentence
    }
    features = pd.DataFrame(data,index=[0])
    return features 

daf = user_input_features()
st.subheader('User Input details')
st.write(daf)
    


df = pd.read_csv("hotel.csv")
        
X = df['sentence']
y = df['Sentiment_Type']

        # Extract Feature With TfidfVectorizer
tv=TfidfVectorizer()
X=tv.fit_transform(df['sentence'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
       
clf = LinearSVC()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)
newadd = sentence
new = [newadd]
vect = tv.transform(new).toarray()
prediction = clf.predict(vect)

st.subheader('Predicted Result')
st.write("Positive Sentiment" if prediction== 1 else "Negative Sentiment")
 

df_idf = df
from stop_words import get_stop_words
stop_words = list(get_stop_words('en'))
docs=df_idf['sentence'].tolist()

cv = CountVectorizer(max_df=0.85,max_features=400000)
word_count_vector=cv.fit_transform(docs)

tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)

df_test = daf
docs_test=df_test['sentence'].tolist()

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

sorted_items=sort_coo(tf_idf_vector.tocoo())
keywords=extract_topn_from_vector(feature_names,sorted_items,10)

st.subheader('Key Words : ')
st.write(keywords)
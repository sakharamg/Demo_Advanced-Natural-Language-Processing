import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Conv1D, GlobalMaxPooling1D
from nltk.tokenize import TreebankWordTokenizer
from gensim.models.keyedvectors import KeyedVectors
import pickle
maxlen = 400
batch_size = 128
embedding_dims = 300
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 10
n_samples=50000
from nltk.stem.porter import * 
stemmer = PorterStemmer() 
            
if 'word_vectors' not in st.session_state:
    st.session_state.word_vectors=KeyedVectors.load_word2vec_format('sentiment_w2v/models/GoogleNews-vectors-negative300.bin', binary=True)
    print("\n\n***WORD VECTORS LOADED FROM W2V***\n\n")

if 'w2v_fb' not in st.session_state:
    st.session_state.w2v_fb=KeyedVectors.load_word2vec_format('sentiment_w2v/models/model.bin', binary=True)
    print("\n\n***FREEBASE WORD VEC LOADED FROM W2V***\n\n")

if 'model' not in st.session_state:
    st.session_state.model = keras.models.load_model('sentiment_w2v/models/w2v_lstm_amazonreviews')
    print("\n\n***LSTM FOR SENTIMENT ANALYSIS LOADED***\n\n")

if 'load_bow_vectorizer' not in st.session_state:
    st.session_state.load_bow_vectorizer = pickle.load(open("sentiment_w2v/models/bow_vectorizer.sav", 'rb'))
    print("\n\n***BOW VECTORIZER LOADED***\n\n")

if 'load_lreg' not in st.session_state:
    st.session_state.load_lreg = pickle.load(open("sentiment_w2v/models/bow_senti.sav", 'rb'))
    print("\n\n***TRAINED LOG REGRESSION LOADED***\n\n")

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt
def tokenize_and_vectorize(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    expected = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[0])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(st.session_state.word_vectors[token])
                
            except KeyError:
                pass # no matching token in the Google w2v vocab
            
        vectorized_data.append(sample_vecs)
    
    return vectorized_data
def pad_trunc(data, maxlen):
    """
    for a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
        
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            #append the appropriate number 0 vectors to the list
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)         
        else:
            temp = sample
        new_data.append(temp)
    return new_data
st.title('Word2Vec Demo')
tab1, tab2 = st.tabs(["Sentiment Analysis"," "])
with tab1:
    st.header("**Sentiment Analysis**")
    with st.form("analogy_form"):
        # st.write("**Sentiment Analysis**")
        sent = st.text_input('Input', 'Today is a great day.')
        # Every form must have a submit button.
        submitted = st.form_submit_button("Find Sentiment")
        if submitted:
            vec_list = tokenize_and_vectorize([(sent, 1)])
            test_vec_list = pad_trunc(vec_list, maxlen)
            test_vec = np.reshape(test_vec_list, (len(test_vec_list), maxlen, embedding_dims))
            score=st.session_state.model.predict(test_vec)[0][0]
            if score>0.5:
                st.write("Sentiment using Word2Vec: :thumbsup: ",score)
            else:
                st.write("Sentiment using Word2Vec: :thumbsdown: ",score)
            
            sent=pd.DataFrame({"tweet":[sent]})
            sent["tweet"]= np.vectorize(remove_pattern)(sent["tweet"], "@[\w]*") 
            sent["tweet"] = sent["tweet"].str.replace("[^a-zA-Z#]", " ")
            sent["tweet"] = sent["tweet"].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))
            tokenized_tweet = sent["tweet"].apply(lambda x: x.split())
            tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
            for i in range(len(tokenized_tweet)):
                tokenized_tweet[i] = ' '.join(tokenized_tweet[i])    
            sent["tweet"] = tokenized_tweet
            from sklearn.feature_extraction.text import CountVectorizer 
            inf_bow = st.session_state.load_bow_vectorizer.transform(sent["tweet"])
            prediction = st.session_state.load_lreg.predict_proba(inf_bow)
            bow_score=prediction[:,1][0]
            if bow_score>0.5:
                st.write("Sentiment using Bag of Words: :thumbsup: ",bow_score)
            else:
                st.write("Sentiment using Bag of Words: :thumbsdown: ",bow_score)

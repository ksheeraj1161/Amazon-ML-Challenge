import os
import re
import glob
import pandas as pd
import numpy as np
import collections
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.feature_selection import chi2.SelectKBest
from sklearn.svm import SVC
import csv

files=glob.glob(os.path.join('data','*'))
train=pd.read_scv(r"c:\Users\Mini|dataset\train.csv",escapechar="\\",quoting=csv.QUOTE_NONE)
test=pd.read_scv(r"c:\Users\Mini|dataset\test.csv",escapechar="\\",quoting=csv.QUOTE_NONE)

train.head()
test.head()

#feature extraction
tfdif=TfidVectorizer(ngram_range=(1,2))
tfidf.fit(train['TITLE'])
X_sparse=tfidf.transformation(['TITLE'])

clf_svc=SVC(degree=3,gamma=1,coef0=1,class_weight=None,max_iter=-1,)

from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

x=stemSentence(sentence)

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

numpy.random.seed(7)

top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
data={'Index':PRODUCT,'Value':BROWSE_NODE_ID)
[8lj = pd.DataFrame(data)
[8lj.to_csv('submission87.csv', index = False)

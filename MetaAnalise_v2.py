# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:20:16 2019

@author: Lab-CEPOF
"""

from __future__ import print_function

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.chdir('G:\Meu Drive\GoogleDrive20180110\DoutoradoUSP\DATA_Miriam')
a=os.listdir()

import scipy.io as spy

mat=spy.loadmat(a[8])


mdata = mat['papers']  # variable in mat file
mdtype = mdata.dtype  # dtypes of structures are "unsized objects"
# * SciPy reads in structures as structured NumPy arrays of dtype object
# * The size of the array is the size of the structure array, not the number
#   elements in any particular field. The shape defaults to 2-dimensional.
# * For convenience make a dictionary of the data using the names from dtypes
# * Since the structure has only one element, but is 2-D, index it at [0, 0]
ndata = {n: mdata[n][0, 0] for n in mdtype.names}


# Reconstruct the columns of the data table from just the time series
# Use the number of intervals to test if a field is a column or metadata
columns = [n for n, v in ndata.items()]
# now make a data frame, setting the time stamps as the index


    

L=[[],[],[],[],[],[]]


for jj in range(np.shape(L)[0]):
    temp2=[]
    for ii in range(np.shape(mdata)[1]):
        try:
            temp=mdata.take(ii)[jj][0]
        except:
            temp=''
    
        temp2.append(str(temp))
    L[jj]=temp2





df=pd.DataFrame(L,columns)
df=df.T
#%% Limpando dataframe - só journal article
# Testando Pandas

#GRoupby
teste=df.groupby('ArticleType').count()


df=df.loc[df['ArticleType'].str.contains("Journal Article")]
df=df.loc[~df['ArticleType'].str.contains("Case")]
df=df.loc[~df['ArticleType'].str.contains("Comment")]
df=df.loc[~df['ArticleType'].str.contains("Review")]
df=df.loc[~df['ArticleType'].str.contains("Guideline")]
df=df.loc[df['PublicationType']=='PubmedArticle']

df=df.loc[~df['title'].str.contains("Review")]
df=df.loc[~df['title'].str.contains("pilot")]
df=df.loc[~df['title'].str.contains("in vivo")]
df=df.loc[~df['title'].str.contains("in vitro")]
df=df.loc[~df['title'].str.contains("ex vivo")]
df=df.loc[~df['title'].str.contains("mouse")]
df=df.loc[~df['title'].str.contains("mice")]
df=df.loc[~df['title'].str.contains("murine")]
df=df.loc[~df['title'].str.contains("cells")]
df=df.loc[~df['title'].str.contains("cell line")]
df=df.loc[~df['title'].str.contains("cell culture")]
df=df.loc[~df['title'].str.contains("commentary on")]
df=df.loc[~df['title'].str.contains("response to")]
df=df.loc[~df['title'].str.contains("retrospective")]


df['KeyWords']=np.zeros(df.shape[0])
inde=df.loc[df['title'].str.contains("surgery")].index
df['KeyWords'][inde.values]=1
inde=df.loc[df['title'].str.contains("randomized")].index
df['KeyWords'][inde.values]=2


teste=df.groupby('PublicationType').count()
#LOC iLOC


#DROPNA or FILL NA

#
#xml = objectify.parse(a[2])
#root = xml.getroot()
#root.tag
#root.attrib




#%% importa tudo
""" Agora começa a rede o tokizer"""

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
import re

from nltk.corpus import stopwords
max_features=5000
max_sequence_length=500


def clean_str(string):
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    cleanr = re.compile('<.*?>')

    string = re.sub(r'\d+', '', string)
    string = re.sub(cleanr, '', string)
    string = re.sub("'", '', string)
    string = re.sub(r'\W+', ' ', string)
    string = string.replace('_', '')


    return string.strip().lower()




def prepare_data(data):    
    # data= pandas.DataFrame = df neste caso
    df['clean_abstract'] = df['abstract'].apply(lambda x: clean_str(x))

    
    stop_words = set(stopwords.words('english'))
    text = []
    for row in data['clean_abstract'].values:
        word_list = text_to_word_sequence(row)
        no_stop_words = [w for w in word_list if not w in stop_words]
        no_stop_words = " ".join(no_stop_words)
        text.append(no_stop_words)


    tk = Tokenizer(num_words=max_features, split=' ')

    tk.fit_on_texts(text)
    X = tk.texts_to_sequences(text)  
    
    X = pad_sequences(X, maxlen=max_sequence_length)
    #X = pad_sequences(X)

    word_index = tk.word_index
    Y = pd.get_dummies(data['KeyWords']).values
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 1000)

    return X_train, X_test, Y_train, Y_test, word_index, tk , text


#%% tokenizer

X_train, X_test, Y_train, Y_test, word_index, tk, sentences = prepare_data(df)

# summarize what was learned
print(tk.word_counts)
print(tk.document_count)
print(tk.word_index)
print(tk.word_docs)

vocab_size = len(tk.word_counts)+1
print(vocab_size)

#%%
from keras.models import Model
from keras.layers import Input, Embedding, Reshape, dot, Dense
from keras.preprocessing import sequence
#from keras.optimizers import SGD, RMSprop, Adagrad
#from keras.utils import np_utils, generic_utils
#from keras.models import Sequential
X=np.vstack((X_train,X_test))
X=np.reshape(X,X.shape[0]*X.shape[1])

window_size = 3
vector_dim = 300
epochs = 1000000

valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)





sampling_table = sequence.make_sampling_table(vocab_size)
couples, labels = sequence.skipgrams(X, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

print(couples[:10], labels[:10])




# create some input variables
input_target = Input((1,))
input_context = Input((1,))

embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')


target = embedding(input_target)
target = Reshape((vector_dim, 1))(target)
context = embedding(input_context)
context = Reshape((vector_dim, 1))(context)

# setup a cosine similarity operation which will be output in a secondary model
similarity= dot([target, context],axes=0,normalize=False)
#similarity = merge([target, context], mode='dot', dot_axes=0)



# now perform the dot product operation to get a similarity measure
dot_product= dot([target, context],axes=1,normalize=False)
#dot_product = merge([target, context], mode='dot', dot_axes=1)
dot_product = Reshape((1,))(dot_product)
# add the sigmoid output layer
output = Dense(1, activation='sigmoid')(dot_product)

# create the primary training model
model = Model(input=[input_target, input_context], output=output)
model.compile(loss='binary_crossentropy', optimizer='rmsprop')

# create a secondary validation model to run our similarity checks during training
validation_model = Model(input=[input_target, input_context], output=similarity)



class SimilarityCallback:
    def run_sim(self):
        for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8  # number of nearest neighbors
            sim = self._get_sim(valid_examples[i])
            nearest = (-sim).argsort()[1:top_k + 1]
            log_str = 'Nearest to %s:' % valid_word
            for k in range(top_k):
                close_word = reverse_dictionary[nearest[k]]
                log_str = '%s %s,' % (log_str, close_word)
            print(log_str)

    @staticmethod
    def _get_sim(valid_word_idx):
        sim = np.zeros((vocab_size,))
        in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
        for i in range(vocab_size):
            in_arr1[0,] = valid_word_idx
            in_arr2[0,] = i
            out = validation_model.predict_on_batch([in_arr1, in_arr2])
            sim[i] = out
        return sim
sim_cb = SimilarityCallback()

arr_1 = np.zeros((1,))
arr_2 = np.zeros((1,))
arr_3 = np.zeros((1,))
for cnt in range(epochs):
    idx = np.random.randint(0, len(labels)-1)
    arr_1[0,] = word_target[idx]
    arr_2[0,] = word_context[idx]
    arr_3[0,] = labels[idx]
    loss = model.train_on_batch([arr_1, arr_2], arr_3)
#    if i % 100 == 0:
    print("Iteration {}, loss={}".format(cnt, loss))
    if cnt % 10000 == 0:
        sim_cb.run_sim()

#%% montando os pesos dos embeddings
#
        # convert the wv word vectors into a numpy matrix that is suitable for insertion
# into our TensorFlow and Keras models
embedding_matrix = np.zeros((len(model.wv.vocab), vector_dim))
for i in range(len(model.wv.vocab)):
    embedding_vector = model.wv[model.wv.index2word[i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        
        
#%% após ter os pesos dos embeddings
        alid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# input words - in this case we do sample by sample evaluations of the similarity
valid_word = Input((1,), dtype='int32')
other_word = Input((1,), dtype='int32')
# setup the embedding layer
embeddings = Embedding(input_dim=embedding_matrix.shape[0], output_dim=embedding_matrix.shape[1],
                      weights=[embedding_matrix])
embedded_a = embeddings(valid_word)
embedded_b = embeddings(other_word)
similarity = merge([embedded_a, embedded_b], mode='cos', dot_axes=2)
# create the Keras model
k_model = Model(input=[valid_word, other_word], output=similarity)

def get_sim(valid_word_idx, vocab_size):
    sim = np.zeros((vocab_size,))
    in_arr1 = np.zeros((1,))
        in_arr2 = np.zeros((1,))
    in_arr1[0,] = valid_word_idx
    for i in range(vocab_size):
        in_arr2[0,] = i
        out = k_model.predict_on_batch([in_arr1, in_arr2])
        sim[i] = out
    return sim

# now run the model and get the closest words to the valid examples
for i in range(valid_size):
    valid_word = wv.index2word[valid_examples[i]]
    top_k = 8  # number of nearest neighbors
    sim = get_sim(valid_examples[i], len(wv.vocab))
    nearest = (-sim).argsort()[1:top_k + 1]
    log_str = 'Nearest to %s:' % valid_word
    for k in range(top_k):
        close_word = wv.index2word[nearest[k]]
        log_str = '%s %s,' % (log_str, close_word)
    print(log_str)

        #%%

#encoded_docs = tk.texts_to_matrix(sentences, mode='count')
#print(encoded_docs)



########################################################################


modelkmeans = KMeans(n_clusters=10,max_iter=200)
kmeans=modelkmeans.fit(X_train)
from matplotlib import pyplot as plt
plt.hist(kmeans.labels_.tolist(),bins=10)

encoded_sentence_cluster=kmeans.cluster_centers_

sentence_cluster=tk.sequences_to_texts(encoded_sentence_cluster)
labels=kmeans.labels_
y_pred=kmeans.fit_predict(X_test)

print(sentence_cluster[2])
print(sentence_cluster[8])


#%% 
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
#from keras.datasets import imdb

# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

#print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('x_train shape:', X_train.shape)
print('x_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, Y_test))

import pickle

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


def load_dataset(file, rows=None):
    """
    Return Pandas DataFrame with the specified number of rows
    """
    r = rows or 10

    dataset = pd.read_csv(file, header=None, names=['sentiment', 'text'], nrows=r)
    
    X = dataset['text']
    Y = dataset['sentiment']
    
    """
    Replace yelp dataset polarity
    from 1=negative, 2=positive
    to   0=negative, 1= positive
    """
    Y = Y.replace(1, 0)
    Y = Y.replace(2, 1)

    return X.tolist(), Y.tolist()


def load_processed_dataset(file, maxseq=None, maxword=None, **kwargs):
    

    MAX_SEQUENCE = maxseq or 1000
    MAX_WORDS = maxword or 20000 # 20000 most common words


    rows = kwargs.get('rows', None)
    X_train, Y_train = load_dataset(file, rows)
        
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)
    
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(X_train)
    
    X1_train = pad_sequences(sequences, maxlen=MAX_SEQUENCE)
    # Y1_train = to_categorical(np.asarray(Y_train), 2)
    del X_train
    # del Y_train
    
    return X1_train, np.asarray(Y_train), word_index



def load_embeddings(dim=True, embeddings=True) ->dict:
    """
    Uses GloVe (global vectors)

    Returns:      
       Embeddings(dict) and Dimensions(int) if both dim and embeddings are set to true      
       
       Dimensions(int) if embeddings is false    
       
       Embeddings(dict) if dim is false
    """
    if dim and embeddings:
        return pickle.load(open('glove/glove.twitter.27B.25d.dict.p', 'rb')), 25
    if dim and not embeddings:
        return 25
    if not dim and embeddings:
        return pickle.load(open('glove/glove.twitter.27B.25d.dict.p', 'rb'))

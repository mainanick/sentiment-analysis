import os

import numpy as np
import pandas as pd
from keras.layers import (AveragePooling1D, Conv1D, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D, MaxPool1D)
from keras.models import Sequential

from utils import load_embeddings, load_processed_dataset

TRAIN_FILE_TITLE = 'YELP REVIEWS POLARITY DATASET'
TRAIN_FILE = 'datasets/yelp/train.csv'
TEST_FILE = 'datasets/yelp/test.csv'


POLARITY_LABEL = {'negative': 0, 'positive': 1}

MAX_SEQUENCE = 1000
MAX_WORDS = 20000

def gen_model():
    """
    sequences: Embedding Sequences
    """
    sequences = embedding_layer()

    model = Sequential([
    sequences,
    Conv1D(512, 5, activation='relu'),
    AveragePooling1D(pool_size=5),
    Conv1D(256, 5, activation='relu'),
    AveragePooling1D(pool_size=5),
    Conv1D(128, 5, activation='relu'),
    MaxPool1D(pool_size=5),
    GlobalMaxPooling1D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(len(POLARITY_LABEL), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    
    return model


def embedding_layer():
    EMBEDDINGS, EMBEDDING_DIM = load_embeddings()
    matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM)) # Embedding Matrix
    max_words = min(MAX_WORDS, len(word_index))

    for word, i in word_index.items():
        if i >=MAX_WORDS:
            continue
        vector = EMBEDDINGS.get(word, None)
        if vector is not None:
            matrix[i]=vector
        
    layer = Embedding(MAX_WORDS,
                    EMBEDDING_DIM, 
                    weights=[matrix],
                    input_length=MAX_SEQUENCE,
                    trainable=False
                    )
    
    return layer

if __name__ == "__main__":
    """
    Ignore TensorFlow Warnings:
      TF library wasn't compiled to use SSE instructions,..,but could speed up CPU computations
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    nrows = 5000
    X, Y, word_index = load_processed_dataset(TRAIN_FILE, maxword=MAX_WORDS, rows=nrows, tokenize=True)
    
    split = nrows-int((nrows/10))
    
    X_train = X[:split]
    Y_train = Y[:split]
    
    X_test =  X[split:]
    Y_test = Y[split:]

    
    model = gen_model()
    model.fit(X_train, Y_train, validation_data=[X_test, Y_test], batch_size=256, epochs=10)
    model.save("sentiment.h5")

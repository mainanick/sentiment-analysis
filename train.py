import os
import json
import pickle

import numpy as np
import pandas as pd
from keras.layers import (AveragePooling1D, Conv1D, Dense, Dropout, Embedding,
                          Flatten, GlobalMaxPooling1D, LSTM, MaxPool1D)
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


MAX_SEQUENCE = 1000
MAX_WORDS = 2000
POLARITY_LABEL = {'negative': 0, 'positive': 1}
TRAIN_FILE = '../datasets/umich-sentiment/train.csv'


def load_dataset(file, rows):

    dataset = pd.read_csv(file, nrows=rows)

    X = dataset['text']
    Y = dataset['sentiment']

    return X.tolist(), Y.tolist()


def load_processed_dataset(file, maxseq=None, maxword=None, rows=None, **kwargs):
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils.np_utils import to_categorical

    MAX_SEQ = maxseq or MAX_SEQUENCE
    MAX_WORDS = maxword or 20000

    X_train, Y_train = load_dataset(file, rows)

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(X_train)

    X1_train = pad_sequences(sequences, maxlen=MAX_SEQ)

    Y1_train = to_categorical(np.asarray(Y_train), 2)

    del X_train
    del Y_train

    return X1_train, Y1_train, word_index


def load_embeddings():
    return pickle.load(open('../glove/glove.twitter.27B.25d.dict.p', 'rb')), 25


def embedding_layer(word_index):
    EMBEDDINGS, EMBEDDING_DIM = load_embeddings()

    matrix = np.zeros((MAX_WORDS, EMBEDDING_DIM))  # Embedding Matrix
    max_words = min(MAX_WORDS, len(word_index))

    for word, i in word_index.items():
        if i >= max_words:
            continue
        vector = EMBEDDINGS.get(word, None)
        if vector is not None:
            matrix[i] = vector

    layer = Embedding(MAX_WORDS,
                      EMBEDDING_DIM,
                      weights=[matrix],
                      input_length=MAX_SEQUENCE,
                      trainable=False
                      )

    return layer


def gen_model(sequences):
    model = Sequential([
        sequences,  # sequences: Embedding Sequences
        Conv1D(256, 5, activation='relu'),
        AveragePooling1D(pool_size=5),
        Conv1D(128, 5, activation='relu'),
        AveragePooling1D(pool_size=5),
        Conv1D(64, 5, activation='relu'),
        MaxPool1D(pool_size=5),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(POLARITY_LABEL), activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['acc'])

    return model


X, Y, word_index = load_processed_dataset(
    TRAIN_FILE, maxword=MAX_WORDS, rows=10, tokenize=True)


with open('../model/dict.json', 'w') as dictionary_file:
    json.dump(word_index, dictionary_file)

sequences = embedding_layer(word_index)


model = gen_model(sequences)

# print(sequences.get_weights()[0][87])
model.fit(X, Y, validation_split=0.1, batch_size=2, epochs=1)

# model_json = model.to_json()
# with open("../model/model.json", "w") as json_file:
#     json_file.write(model_json)

# model.save_weights("../model/weights.h5")
# model.save("../model/sentiment.h5")

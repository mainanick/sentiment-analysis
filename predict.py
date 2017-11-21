import os
import json

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import keras.preprocessing.text as txt
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask("sentiment")


def load_model():
    json_file = open("model/model_yelp.json", 'r')
    json_model = json_file.read()
    json_file.close()

    model = model_from_json(json_model)
    model.load_weights('model/weights_yelp.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['acc'])

    return model, tf.get_default_graph()


def process_sentence(sentence):

    sentence = txt.text_to_word_sequence(sentence.lower())

    with open('model/dict_yelp.json', 'r') as dictionary_file:
        word_index = json.load(dictionary_file)

    word_indices = []

    for word in sentence:
        if word not in word_index:
            continue    
        word_indices.append(word_index[word])

    return [word_indices]


def process(sentence):
    MAX_SEQUENCE = 1000
    MAX_WORDS = 20000

    sequences = process_sentence(sentence)

    tokenizer = Tokenizer(num_words=MAX_WORDS)

    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE)
    print(X)
    return X


model, graph = load_model()


@app.route('/predict', methods=['POST'])
def predict():
    sentence = request.get_json()['sentence']

    X = process(sentence)

    with graph.as_default():
        pred = model.predict(X)

    print(pred)
    POLARITY_LABEL = {0: 'negative', 1: 'positive'}
    sentiment = POLARITY_LABEL[np.argmax(pred)]

    response = {"sentence": sentence, "sentiment": sentiment}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

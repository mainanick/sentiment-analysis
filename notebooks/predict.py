
# coding: utf-8

# In[1]:

import os

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical


# In[2]:

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

app = Flask("sentiment")


# In[3]:

def load_model():
    json_file = open("../model/model.json", 'r')    
    json_model = json_file.read()    
    json_file.close()
    
    model = model_from_json(json_model)
    model.load_weights('../model/weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    
    return model, tf.get_default_graph()


# In[4]:

def process(sentence):
    MAX_SEQUENCE = 1000
    MAX_WORDS = 20000
    
    sentence = [sentence]
    
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(sentence)
    
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentence)
    
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE)

    return X


# In[5]:

model, graph = load_model()

@app.route('/predict', methods=['POST'])
def predict():

    sentence = request.get_json()['sentence']
    X = process(sentence)
    with graph.as_default():
        pred = model.predict(X)
    
    response = {"sentiment": str(pred), "sentence": sentence}

    return jsonify(response)

if __name__ == '__main__':
	app.run(debug=True)





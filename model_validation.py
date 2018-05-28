import pickle
from glob import glob
from collections import OrderedDict
import xmltodict
import keras
from api_model_definition import losses
import numpy as np
import sys
from keras.utils.np_utils import to_categorical
from keras.layers import Input
from cnn import build_cnn_return_preds
from cnn_w_attention import attn_build_cnn_return_preds, losses
from pprint import pprint
from keras.optimizers import adam
from keras.models import Model
import csv
INPUT_SHAPE = (40,)
dictionaries = pickle.load(open('./dictionaries.p', 'rb'))
MAX_SEQUENCE_LEN = 40
MIN_SEQUENCE_LEN = 5

def create_test_data():
    with open('./validation.csv', 'r') as f:
        reader = csv.DictReader(f)
        next(reader)
        dataset = []
        for row in reader:
            text = row['text']
            author = row['author']
            dataset.append((author, text[:MAX_SEQUENCE_LEN]))
    return dataset

def translate_words_to_indices(data, dictionary, author_dictionary):
    translated = []
    for i, x in enumerate(dataset):
        author, text = x
        tmp = list(map(lambda doc: int(dictionary.get(doc, 0)), text.lower().split(' ')))
        if len(tmp) < MIN_SEQUENCE_LEN:
            continue
        elif len(tmp) > MAX_SEQUENCE_LEN:
            tmp = tmp[:MAX_SEQUENCE_LEN]
        translated.append((author_dictionary[author], tmp))
    return translated

def pad_to_max_length(data, pad_token):
    for doc in data:
        if len(doc[1]) == MAX_SEQUENCE_LEN:
            continue
        doc[1].extend([pad_token]*(MAX_SEQUENCE_LEN-len(doc[1])))

def train_model(build_fn, valid_x, valid_y, epochs, n_layers=1):
    print(epochs, n_layers)
    dicts = pickle.load(open('./dictionaries.p', 'rb'))
    training_set = pickle.load(open('./train_dataset.p', 'rb'))
    inputs = Input(shape=INPUT_SHAPE)
    preds, maps = build_fn(inputs, n_layers)
    model = Model(inputs=[inputs], outputs=preds)
    optimizer = adam()
    if not maps:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss=losses(maps), metrics=['accuracy'])
    inputs = []
    labels = []
    for x in training_set:
        inputs.append(x['text'])
        labels.append(x['author'])
    inputs = np.asarray(inputs)
    labels = np.expand_dims(to_categorical(np.asarray(labels)), 1)
    model.summary()
    model.fit(inputs, labels, epochs=epochs, batch_size=100, shuffle='batch', validation_data=(valid_x, valid_y))
    model.save('./saved_model')


dataset = create_test_data()
dataset = translate_words_to_indices(dataset, dictionaries['dict'], dictionaries['authors'])
pad_to_max_length(dataset, 0)
y_set, x_set = zip(*dataset)
x_set = np.asarray(x_set)
y_set = np.asarray(y_set)
y_set = np.expand_dims(to_categorical(y_set, 3), 1)

action = None
if len(sys.argv) < 2:
    action = 'vanilla'
elif sys.argv[1] == 'attention':
    action = 'attention'
elif sys.argv[1] == 'fatt':
    action = 'fatt'
elif sys.argv[1] == 'eval':
    if len(sys.argv) < 3:
        print('Must supply model file.')
        exit()
    action = 'eval'
else:
    action = 'vanilla'

if action == 'eval':
    loss = losses([])
    model = keras.models.load_model(sys.argv[2], custom_objects={'combined_loss': loss})
    metrics = model.evaluate(x_set, y_set, batch_size=100, verbose=1)
    pprint(metrics)
else:
    if action == 'fatt':
        print('Training Fasttext Attention Model')
        train_model(build_cnn_return_preds, x_set, y_set, int(sys.argv[2]), int(sys.argv[3]))
    elif action == 'attention':
        print('Training Attention Model')
        train_model(attn_build_cnn_return_preds, x_set, y_set, int(sys.argv[2]), int(sys.argv[3]))
    else:
        print('Training Normal CNN Model')
        train_model(build_cnn_return_preds, x_set, y_set, int(sys.argv[2]), int(sys.argv[3]))




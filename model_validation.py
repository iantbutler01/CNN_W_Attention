import pickle
from glob import glob
from collections import OrderedDict
import xmltodict
import keras
from api_model_definition import losses
import numpy as np
from keras.utils.np_utils import to_categorical
from pprint import pprint
files = glob('./pan11/*')
files = filter(lambda fl: fl.find('LargeValid') != -1 or fl.find('GroundTruthLargeValid') != -1, files)
files = filter(lambda fl: fl.find('+') == -1, files)

dictionaries = pickle.load(open('./dictionaries.p', 'rb'))
MAX_SEQUENCE_LEN = 40
MIN_SEQUENCE_LEN = 5

def create_test_data(x_file, y_file):
    with open(x_file, 'r') as f:
        label_rows = xmltodict.parse(f.read())['results']['text']

    with open(y_file, 'r') as f:
        data_rows = xmltodict.parse(f.read())['testing']['text']
    dataset = {}
    for x, y in zip(data_rows, label_rows):
        body = x['body']
        if type(body) == OrderedDict:
            if 'OMNI' in body.keys() or 'OMNIPAB' in body.keys():
                continue
            text = body['#text'][:MAX_SEQUENCE_LEN]
        else:
            text = body
        author = y['author']['@id']
        dataset[author] = text[:MAX_SEQUENCE_LEN]
    return dataset

def translate_words_to_indices(data, dictionary, author_dictionary):
    translated = []
    for i, x in enumerate(data.items()):
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

dataset = create_test_data(*list(files))
dataset = translate_words_to_indices(dataset, dictionaries['dict'], dictionaries['authors'])
pad_to_max_length(dataset, 0)
y_set, x_set = zip(*dataset)
x_set = np.asarray(x_set)
y_set = np.asarray(y_set)
y_set = np.expand_dims(to_categorical(y_set, 70), 1)
print(y_set)
loss = losses([])
model = keras.models.load_model(f'./saved_model', custom_objects={'combined_loss': loss})
metrics = model.evaluate(x_set, y_set, batch_size=30, verbose=1)
pprint(metrics)




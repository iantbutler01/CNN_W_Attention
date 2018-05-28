import xmltodict
from glob import glob
import sys
import string
import re
import itertools
import json
import pickle
import csv
MAX_SEQUENCE_LEN = 40
MIN_SEQUENCE_LEN = 5

def reformat(files):
    dataset = []
    with open('./train.csv', 'r+') as f:
        reader = csv.DictReader(f)
        next(reader)
        for row in reader:
            text = row['text']
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            text = re.sub(r"\t+|\n+|'+",'',text)
            dataset.append({ 'author': row['author'], 'text': text})
    return dataset

def build_dictionaries(data):
    words = []
    authors = []
    for sample in data:
        words.extend(sample['text'].split(' '))
        authors.append(sample['author'])
    words = ['</p>'] + list(set(words))
    authors = list(set(authors)) #unique the list
    d_authors = dict(enumerate(authors)).items() # enumerate [(index, author)] and convert to dict
    d_authors = { v: k for k, v in d_authors } # reverse the dictionary, { index: author } to { author: index }
    word_dict = { v: k for k, v in dict(enumerate(words)).items()}
    return { 'r_dict': words, 'dict': word_dict, 'authors': d_authors, 'authors_rdict': authors}

def translate_words_to_indices(data, dictionary, author_dictionary):
    translated = []
    for i, x in enumerate(data):
        tmp = list(map(lambda doc: dictionary[doc], x['text'].split(' ')))
        if len(tmp) < MIN_SEQUENCE_LEN:
            continue
        elif len(tmp) > MAX_SEQUENCE_LEN:
            tmp = tmp[:MAX_SEQUENCE_LEN]
        translated.append({ 'author': author_dictionary[x['author']], 'text': tmp})
    return translated

def pad_to_max_length(data, pad_token):
    for doc in data:
        if len(doc['text']) == MAX_SEQUENCE_LEN:
            continue
        doc['text'].extend([pad_token]*(MAX_SEQUENCE_LEN-len(doc['text'])))

files = glob('./pan11/**')

if len(sys.argv) < 2:
    raise AttributeError('Must provide document type to process, Train or Valid')
dataset = sys.argv[1]
if dataset != 'Train' and dataset != 'Valid':
    raise AttributeError("Argument must be Train or Valid")

data = reformat(files)
dicts = build_dictionaries(data)
data = translate_words_to_indices(data, dicts['dict'], dicts['authors'])
pad_to_max_length(data, 0)
pickle.dump(dicts, open(f'{dataset.lower()}_dictionaries.p', 'wb'))
pickle.dump(data, open(f'{dataset.lower()}_dataset.p', 'wb'))

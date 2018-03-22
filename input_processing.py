import xmltodict
from glob import glob
import sys
import string
import re
import itertools
import json
import pickle
MAX_SEQUENCE_LEN = 400
MIN_SEQUENCE_LEN = 5

def reformat(files, dataset):
    files = list(filter(lambda fn: fn.find(f'Large{dataset}') != -1 and fn.find('GroundTruth') == -1, files))
    data_by_file = {}
    for x in files:
        data_by_file[x] = []
        print(x)
        with open(x, 'r+') as f:
            new_obj = xmltodict.parse(f.read())
            for dat in new_obj['training']['text']:
                if type(dat['body']) != str:
                    if not '#text' in dat['body'].keys():
                        continue
                    text = dat['body']['#text']
                else:
                    text = dat['body']
                text = text.translate(str.maketrans('', '', string.punctuation)).lower()
                text = re.sub(r"\t+|\n+|'+",'',text)
                data_by_file[x].append( { 'author': dat['author']['@id'], 'text': text})
    return data_by_file
def build_dictionaries(data):
    def key_to_dict(items, key):
        return itertools.chain.from_iterable(map(lambda x: x[key].split(' '), items))
    words = []
    authors = []
    for _,v in data.items():
        factor = 1000
        index = 0
        while True:
            end = len(v) - (len(v) % factor) - 1
            window = v[index:end]
            words.extend(key_to_dict(window, 'text'))
            authors.extend(key_to_dict(window, 'author'))
            index = index + factor
            factor = factor * 4
            if factor > len(v):
                window = v[end:]
                words.extend(key_to_dict(window, 'text'))
                authors.extend(key_to_dict(window, 'author'))
                break
    words = ['</p>'] + list(set(words))
    authors = list(set(authors)) #unique the list
    d_authors = dict(enumerate(authors)).items() # enumerate [(index, author)] and convert to dict
    d_authors = { v: k for k, v in d_authors } # reverse the dictionary, { index: author } to { author: index }
    word_dict = { v: k for k, v in dict(enumerate(words)).items()}
    return { 'r_dict': words, 'dict': word_dict, 'authors': d_authors, 'authors_rdict': authors}

def translate_words_to_indices(data, dictionary, author_dictionary):
    new_data = {}
    for k,v in data.items():
        translated = []
        for i, x in enumerate(v):
            tmp = list(map(lambda doc: dictionary[doc], x['text'].split(' ')))
            if len(tmp) < MIN_SEQUENCE_LEN:
                continue
            elif len(tmp) > MAX_SEQUENCE_LEN:
                tmp = tmp[:MAX_SEQUENCE_LEN]
            translated.append({ 'author': author_dictionary[x['author']], 'text': tmp})
    new_data[k] = translated
    return new_data

def pad_to_max_length(data, pad_token):
    for _,v in data.items():
        for doc in v:
            if len(doc['text']) == MAX_SEQUENCE_LEN:
                continue
            doc['text'].extend([pad_token]*(MAX_SEQUENCE_LEN-len(doc['text'])))

files = glob('./pan11/**')

if len(sys.argv) < 2:
    raise AttributeError('Must provide document type to process, Train or Valid')
dataset = sys.argv[1]
if dataset != 'Train' and dataset != 'Valid':
    raise AttributeError("Argument must be Train or Valid")

data = reformat(files, dataset)
dicts = build_dictionaries(data)
data = translate_words_to_indices(data, dicts['dict'], dicts['authors'])
pad_to_max_length(data, 0)
pickle.dump(dicts, open(f'{dataset.lower()}_dictionaries.p', 'wb'))
pickle.dump(data, open(f'{dataset.lower()}_dataset.p', 'wb'))

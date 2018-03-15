import xmltodict
from glob import glob
import sys
import string
import re
import itertools
import json
import pickle


def reformat(files, dataset):
    files = list(filter(lambda fn: fn.find(f'Large{dataset}') != -1 and fn.find('GroundTruth') == -1, files))
    data_by_file = {}
    for x in files:
        data_by_file[x] = []
        with open(x, 'r+') as f:
            new_obj = xmltodict.parse(f.read())
            for dat in new_obj['training']['text']:
                if type(dat['body']) != str:
                    if not '#text' in dat['body'].keys():
                        continue
                    text = dat['body']['#text']
                else:
                    text = dat['body']
                if(len(text) > 550):
                    continue
                text = text.translate(str.maketrans('', '', string.punctuation)).lower()
                text = re.sub(r"\t+|\n+|'+",'',text)
                data_by_file[x].append( { 'author': dat['author']['@id'], 'text': text})
    return data_by_file
def build_dictionaries(data):
    def words_to_dict(items):
        return itertools.chain.from_iterable(map(lambda x: x['text'].split(' '), items))
    words = []
    for _,v in data.items():
        factor = 1000
        index = 0
        print(len(v))
        while True:
            end = len(v) - (len(v) % factor) - 1
            window = v[index:end]
            words.extend(words_to_dict(window))
            index = index + factor
            factor = factor * 4
            if factor > len(v):
                window = v[end:]
                words.extend(words_to_dict(window))
                break
    words = list(set(words))
    return { 'r_dict': words, 'dict': { v: k for k, v in dict(enumerate(words)).items()}}

def translate_words_to_indices(data, dictionary):
    for _,v in data.items():
        for x in v:
            x['text'] = list(map(lambda doc: dictionary[doc], x['text'].split(' ')))

def pad_to_max_length(data):
    max_len = []
    for _,v in data.items():
       max_len.append(max(map(lambda x: len(x['text']), v)))
    max_len = max(max_len)
    print(max_len)
    for _,v in data.items():
        for doc in v:
            if len(doc['text']) == max_len:
                continue
            doc['text'].extend([max_len+1]*(max_len-len(doc['text'])))

files = glob('./pan11/**')

if len(sys.argv) < 2:
    raise AttributeError('Must provide document type to process, Train or Valid')
dataset = sys.argv[1]
if dataset != 'Train' and set != 'Test':
    raise AttributeError("Argument must be Train or Valid")

data = reformat(files, dataset)
dicts = build_dictionaries(data)
translate_words_to_indices(data, dicts['dict'])
pad_to_max_length(data)
pickle.dump(dicts, open('dictionaries.p', 'wb'))
pickle.dump(data, open(f'{dataset.lower()}_dataset.p', 'wb'))

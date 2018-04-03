from fasttext import skipgram
from pprint import pprint
import pickle
model = skipgram(
    './fasttext_train_data.txt', 'pan_vectors',
    min_count=1,
    silent=0,
    lr=0.0999,
    ws=200,
    dim=128,
    maxn=12
)

#generate the reamining word vectors that fasttext does not capture the first time and append them
#to the output file
dicts = pickle.load(open('./dictionaries.p', 'rb'))

with open('./pan_vectors.vec', 'r') as f:
    next(f)
    vectorized_words = set(map(lambda vec: vec.split(' ')[0], f))

remaining = set([wd for wd, num in dicts['dict'].items()]) - vectorized_words

with open('./pan_vectors.vec', 'a') as f:
    for wd in list(remaining):
        vec = [wd] + list(map(lambda x: str(x), model[wd]))
        f.write(' '.join(vec))
        f.write(' \n')

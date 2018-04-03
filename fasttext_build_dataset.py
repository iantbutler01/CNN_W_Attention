import pickle
from pprint import pprint

training_set = pickle.load(open('./train_dataset.p', 'rb'))
dictionaries = pickle.load(open('./dictionaries.p', 'rb'))
reverse_dictionary = dictionaries.get('r_dict')
data = []
for _, dataset in training_set.items():
    for sample in dataset:
        filtered_sample = filter(lambda wd: reverse_dictionary[wd] != '</p>', sample.get('text'))
        text = ' '.join(map(lambda wd: reverse_dictionary[wd], filtered_sample))
        print(text)
        data.append(text)
with open('fasttext_train_data.txt', 'w+') as f:
    f.write('\n'.join(data))

